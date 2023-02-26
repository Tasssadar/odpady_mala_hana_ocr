import cv2
import sys
import re
import math
import easyocr
import colour

import numpy as np
import string
import ics
from dataclasses import dataclass
from enum import auto, Enum

from datetime import datetime, timezone, timedelta


# this needs to run only once to load the model into memory
READER = easyocr.Reader(["en"])

CURRENT_YEAR = datetime.now().year


@dataclass
class Rect:
    x: int
    y: int
    w: int
    h: int

    y_group: int = 0

    def dist(self, o: "Rect") -> float:
        return math.dist((self.x, self.y), (o.x, o.y))

    def cutout(self, img_full: cv2.Mat) -> cv2.Mat:
        return img_full.copy()[self.y : self.y + self.h, self.x : self.x + self.w]


MONTHS = [
    "leden",
    "unor",
    "brezen",
    "duben",
    "kveten",
    "cerven",
    "cervenec",
    "srpen",
    "zari",
    "rijen",
    "listopad",
    "prosinec",
]


class TrashType(Enum):
    BIO = auto()
    PLAST_PAPIR = auto()
    KOMUNAL = auto()

    @staticmethod
    def by_lab(lab_value: np.ndarray) -> "TrashType":
        TYPE_BASELINES = [
            (
                TrashType.BIO,
                np.array([53.26302881667688, 152.47026364193746, 144.43408951563458]),
            ),
            (
                TrashType.PLAST_PAPIR,
                np.array([203.92259225922592, 126.86768676867686, 197.4000900090009]),
            ),
            (
                TrashType.PLAST_PAPIR,
                np.array([101.78601997146933, 142.55920114122682, 76.33594864479315]),
            ),
            (
                TrashType.KOMUNAL,
                np.array([44.05375754251234, 128.8134942402633, 126.77290181020297]),
            ),
        ]

        min = 255.0
        for k, v in TYPE_BASELINES:
            d = float(colour.delta_E(lab_value, v))
            if d < min:
                res = k
                min = d
        return res

    def event_name(self) -> str:
        match self:
            case TrashType.BIO:
                return "Biodpad"
            case TrashType.PLAST_PAPIR:
                return "Plast a papír"
            case TrashType.KOMUNAL:
                return "Komunální odpad"


def sort_moth_rects(orig_img_h: int, months: list[Rect]) -> list[Rect]:
    assert len(months) % 2 == 0

    month_stack = list(months)
    ratio_id = 0
    while month_stack:
        m = month_stack.pop()
        m_ratio = m.y / orig_img_h

        closest_ratio = 1
        closest_idx = -1
        for i, o in enumerate(month_stack):
            o_ratio = o.y / orig_img_h
            if abs(o_ratio - m_ratio) < closest_ratio:
                closest_idx = i

        o = month_stack.pop(closest_idx)
        m.y_group = ratio_id
        o.y_group = ratio_id
        ratio_id += 1

    return sorted(found_month_rects, key=lambda r: (r.y_group, r.x))


def decide_half(month_cutout_rgb: cv2.Mat) -> int:
    h, w, _ = month_cutout_rgb.shape
    month_cutout_rgb = month_cutout_rgb[0 : int(h * 0.2), 0:w]

    char_whitelist = string.ascii_letters + string.digits + "_-. "
    recognized_text_list: list[str] = READER.readtext(
        month_cutout_rgb, allowlist=char_whitelist, detail=0
    )
    recognized_text = " ".join(recognized_text_list).lower()

    for i, month in enumerate(MONTHS):
        if month in recognized_text:
            if i <= 5:
                return -1
            else:
                return 1

    m = re.search(r" ([0-9]{2})", recognized_text)
    if m is not None:
        month_num = int(m.group(1))
        if month_num <= 6:
            return -1
        else:
            return 1

    return 0


def find_day_number(day_cutout_rgb: cv2.Mat) -> int:
    hsv_image = cv2.cvtColor(day_cutout_rgb.copy(), cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_image, np.array([0, 0, 230]), np.array([255, 70, 255]))
    mask = cv2.bitwise_not(mask)

    mask = cv2.blur(mask, (2, 2))
    mask = cv2.resize(mask, (600, 400), 0, 0, cv2.INTER_LINEAR)

    mh, mw = mask.shape
    mask = mask.copy()[int(mh * 0.05) : int(mh * 0.95), int(mw * 0.05) : int(mw * 0.95)]

    texts: list[str] = READER.readtext(
        mask, allowlist="0123456789", detail=0, min_size=8
    )

    day = None
    for t in texts:
        try:
            day = int(t, base=10)
            break
        except ValueError:
            pass

    if day is None:
        print("Failed to recognize day number.", texts)

        cv2.imshow("day_num", mask)
        cv2.waitKey()
        sys.exit(1)

    return day


def find_trash_type(day_cutout_rgb: cv2.Mat) -> TrashType:
    ch, cw, _ = day_cutout_rgb.shape
    day_cutout_rgb = day_cutout_rgb.copy()[0 : int(ch / 3), 0:cw]

    day_cutout_hsv = cv2.cvtColor(day_cutout_rgb.copy(), cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(day_cutout_hsv, np.array([0, 0, 230]), np.array([255, 70, 255]))
    mask = cv2.bitwise_not(mask)

    day_cutout_lab = cv2.cvtColor(day_cutout_rgb.copy(), cv2.COLOR_BGR2LAB)
    avg_lab = cv2.mean(day_cutout_lab, mask)
    return TrashType.by_lab(avg_lab[0:3])


def build_event(day_num: int, month_num: int, type: TrashType) -> ics.Event:
    begin = datetime(CURRENT_YEAR, month_num, day_num)
    end = datetime(CURRENT_YEAR, month_num, day_num, 23, 59, 59)

    event = ics.Event(
        name=type.event_name(),
        begin=begin,
        end=end,
        transparent=True,
        # alarms=[ics.DisplayAlarm(trigger=timedelta(hours=4))],
    )
    event.make_all_day()
    return event


def process_month(month_num: int, img_rgb: cv2.Mat) -> list[ics.Event]:
    orig_h, orig_w, _ = img_rgb.shape
    img_rgb = img_rgb.copy()[int(orig_h * 0.2) : orig_h, 0:orig_w]

    img_gray = cv2.cvtColor(img_rgb.copy(), cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(img_gray, 200, 255, 0)
    # cv2.imshow("thres", thresh)
    contours, _ = cv2.findContours(thresh, 1, 2)

    res: list[ics.Event] = []

    img_month_annotations = img_rgb.copy()
    for cnt in contours:
        x1, y1 = cnt[0][0]
        r = Rect(*cv2.boundingRect(cnt))
        ratio_w = r.w / orig_w
        ratio_h = r.h / orig_h

        if ratio_w < 0.08 or ratio_w > 0.2 or ratio_h < 0.08 or ratio_h > 0.2:
            continue

        cv2.drawContours(img_month_annotations, [cnt], -1, (0, 255, 255), 3)
        cv2.rectangle(
            img_month_annotations,
            (r.x, r.y),
            (r.x + r.w, r.y + r.h),
            (255, 0, 0),
            2,
        )

        day_cutout_rgb = img_rgb[r.y : r.y + r.h, r.x : r.x + r.w].copy()
        day_num = find_day_number(day_cutout_rgb)
        day_trash = find_trash_type(day_cutout_rgb)

        cv2.putText(
            img_month_annotations,
            f"{day_num} {day_trash.name}",
            (x1, y1),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

        print(f"    {day_num}.{month_num}. - {day_trash.name}")
        res.append(build_event(day_num, month_num, day_trash))

    # cv2.imshow("frame2", img_month_annotations)
    # cv2.waitKey()

    return res


if __name__ == "__main__":
    calendar = ics.Calendar()

    for img_path in sys.argv[1:]:
        print("Processing", img_path)

        img_rgb = cv2.imread(img_path)
        # img_rgb = cv2.resize(img_rgb, (768, 1024), interpolation=cv2.INTER_CUBIC)

        orig_h, orig_w, _ = img_rgb.shape

        img_gray = cv2.cvtColor(img_rgb.copy(), cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(img_gray, 80, 255, 0)
        # cv2.imshow('thres', thresh)
        contours, hierarchy = cv2.findContours(thresh, 1, 2)

        found_month_rects: list[Rect] = []
        img_month_annotations = img_rgb.copy()
        for cnt in contours:
            x1, y1 = cnt[0][0]
            approx = cv2.approxPolyDP(cnt, 0.01 * cv2.arcLength(cnt, True), True)
            if len(approx) >= 4 and len(approx) <= 8:
                r = Rect(*cv2.boundingRect(cnt))
                ratio_w = r.w / orig_w
                ratio_h = r.h / orig_h
                if not (
                    (ratio_w > 0.35 and ratio_w < 0.45)
                    and (ratio_h > 0.20 and ratio_h < 0.28)
                ):
                    continue

                if any(True for ex in found_month_rects if ex.dist(r) / orig_w < 0.25):
                    continue

                found_month_rects.append(r)

                cv2.drawContours(img_month_annotations, [cnt], -1, (0, 255, 255), 3)
                cv2.rectangle(
                    img_month_annotations,
                    (r.x, r.y),
                    (r.x + r.w, r.y + r.h),
                    (255, 0, 0),
                    2,
                )
                cv2.putText(
                    img_month_annotations,
                    str(len(approx)),
                    (x1, y1),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

        if len(found_month_rects) != 6:
            print("Found %d months instead of 6!" % len(found_month_rects))
            cv2.imshow("frame2", img_month_annotations)
            cv2.waitKey()
            sys.exit(1)

        found_month_rects = sort_moth_rects(orig_h, found_month_rects)

        half_score = 0
        for r in found_month_rects:
            half_score += decide_half(r.cutout(img_rgb))

        if half_score == 0:
            print("Failed to determine which half of the year this image is.")
            sys.exit(1)

        half_offset = 1 if half_score < 0 else 7

        for idx, r in enumerate(found_month_rects):
            print(
                f"Processing month {half_offset + idx} {MONTHS[half_offset + idx - 1]}"
            )
            events = process_month(half_offset + idx, r.cutout(img_rgb))
            calendar.events.update(events)

    with open("odpad_kalendar_%d.ics" % CURRENT_YEAR, "w") as f:
        f.writelines(calendar.serialize_iter())
    # cv2.waitKey()

    cv2.destroyAllWindows()
