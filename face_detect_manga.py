import cv2
from face_d_api_class_client import AnimeFaceDet
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from io import BytesIO
import mojimoji
from face_d_api_class import AnimeFaceDetect
import pandas as pd

AF = AnimeFaceDetect()

small_chars = "ぁぃぅぇぉゃゅょっァィゥェォャュョッ"  # 小さい文字
numbers = "０１２３４５６７８９0123456789"
alphabets = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyzＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺａｂｃｄｅｆｇｈｉｊｋｌｍｎｏｐｑｒｓｔｕｖｗｘｙｚ"
punctuation = "。、．，"
long_char = "|"

(
    bubble_x1,
    bubble_x2,
    bubble_y1,
    bubble_y2,
    bubble_width1,
    bubble_width2,
    bubble_height1,
    bubble_height2,
) = (0, 0, 0, 0, 0, 0, 0, 0)


# 句読点や小さい文字を判定する関数
def special_char_offset(char, font_size):

    if char in small_chars:
        return (font_size // 8, -font_size // 6)
    elif char in punctuation:
        return (font_size // 1.5, -font_size // 1.5)
    elif char in numbers:
        return (font_size // 6, 0)
    elif char in long_char:
        return (font_size // 3, 0)
    elif char in alphabets:
        return (font_size // 6, 0)
    elif char == "・":
        return (font_size // 4, 0)
    else:
        return (0, 0)


def put_speech_bubble_on_image(imagefile, scr):
    results = []
    rec = False
    text = (
        mojimoji.han_to_zen(scr)
        .replace("，", "、")
        .replace("．", "。")
        .replace("…", "\u22ee")
        .replace("「", "\ufe41")
        .replace("」", "\ufe42")
        .replace("『", "\ufe43")
        .replace("』", "\ufe44")
        .replace("（", "\ufe35")
        .replace("）", "\ufe36")
        .replace("～", "\u2307")
    )

    img_data = cv2.imread(imagefile)
    confidence_level = 0.5
    dnum, img_data, predict_bbox, pre_dict_label_index, scores = AF.face_det(
        img_data, confidence_level
    )

    print(
        "dnum=",
        dnum,
        "bbox=",
        predict_bbox,
        "label=",
        pre_dict_label_index,
        "score=",
        scores,
    )

    # 画像読み込みと描画処理の準備
    image = Image.open(imagefile)
    draw = ImageDraw.Draw(image)

    # 顔領域バウンディングボックスの各座標用変数を定義
    x1, y1, x2, y2 = map(int, predict_bbox[0])
    face_center_x = (x1 + x2) // 2
    face_center_y = (y1 + y2) // 2

    # 吹き出しの幅と高さを顔の大きさに基づいて調整
    face_width = x2 - x1
    bubble_width = int((image.width - face_width) / 2)
    bubble_height = int(image.height * 1)

    # 吹き出し用フォントの定義
    font_size = 50
    font = ImageFont.truetype("GenEiAntiquePv5-M.ttf", font_size)
    # font = ImageFont.truetype("NitalagoRuikaMono-04.TTF", font_size)

    # 1行当たりに入る文字数
    max_lines = 15

    # 吹き出しに入る行数
    max_lines_per_bubble = 3

    current_line = []
    lines = []
    cr_flag = False

    # 各文字を縦に描画し、はみ出した場合は改行
    for char in text:
        if cr_flag:
            if (
                not (char in punctuation or char in small_chars or char in long_char)
                or len(lines) == max_lines_per_bubble - 1
            ):
                if (
                    char == "、"
                    or char == "。"
                    or char == "！"
                    or char == "？"
                    or char == "\u22ee"
                ):
                    current_line.append(char)
                    lines.append("".join(current_line))
                    current_line = []
                    cr_flag = False
                else:
                    lines.append("".join(current_line))
                    current_line = []
                    current_line.append(char)
                    cr_flag = False
            else:
                current_line.append(char)

        else:
            # 縦書きで1列に収まるかチェック（文字の行数で調整）
            if (
                len(current_line) >= max_lines
                or (
                    len(lines) == max_lines_per_bubble - 1
                    and (char == "。" or char == "！" or char == "？" or char == "、")
                )
                or char == "\n"
            ):
                cr_flag = True

            if char != "\n":
                current_line.append(char)

    # 最後の行を追加
    if current_line:
        lines.append("".join(current_line))

    # 吹き出し枠の描画処理
    if image.width - x2 >= x1 and text != "":
        bubble_x1 = x2 + font_size // 2
        bubble_y1 = 0
        draw.ellipse(
            [
                (bubble_x1, bubble_y1),
                (bubble_x1 + bubble_width, bubble_y1 + bubble_height),
            ],
            outline="black",
            width=5,
            fill="white",
        )
        draw.polygon(
            (
                (bubble_x1 + 7, (bubble_y1 + bubble_height) // 2),
                (x2 + ((bubble_x1 - x2) // 2), y2),
                (bubble_x1 + 7, ((bubble_y1 + bubble_height) // 2) - font_size),
            ),
            fill=(255, 255, 255),
        )
        draw.line(
            (
                (bubble_x1 + 5, (bubble_y1 + bubble_height) // 2),
                (x2 + ((bubble_x1 - x2) // 2), y2),
                (bubble_x1 + 5, ((bubble_y1 + bubble_height) // 2) - font_size),
            ),
            fill=(0, 0, 0),
            width=6,
        )

        if len(lines) > max_lines_per_bubble and x1 > bubble_width - font_size:
            bubble_x2 = x1 - bubble_width - font_size
            bubble_y2 = 0
            draw.ellipse(
                [
                    (bubble_x2, bubble_y2),
                    (bubble_x2 + bubble_width, bubble_y2 + bubble_height),
                ],
                outline="black",
                width=5,
                fill="white",
            )
            draw.polygon(
                (
                    (
                        bubble_x2 + bubble_width - 7,
                        (bubble_y2 + bubble_height) // 2,
                    ),
                    (
                        bubble_x2
                        + bubble_width
                        + ((x1 - (bubble_x2 + bubble_width)) // 2),
                        y2,
                    ),
                    (
                        bubble_x2 + bubble_width - 7,
                        ((bubble_y2 + bubble_height) // 2) - font_size,
                    ),
                ),
                fill=(255, 255, 255),
            )
            draw.line(
                (
                    (
                        bubble_x2 + bubble_width - 4,
                        (bubble_y2 + bubble_height) // 2,
                    ),
                    (
                        bubble_x2
                        + bubble_width
                        + ((x1 - (bubble_x2 + bubble_width)) // 2),
                        y2,
                    ),
                    (
                        bubble_x2 + bubble_width - 5,
                        ((bubble_y2 + bubble_height) // 2) - font_size,
                    ),
                ),
                fill=(0, 0, 0),
                width=6,
            )

        elif len(lines) > max_lines_per_bubble:
            rec = True
            results.append(
                put_speech_bubble_on_image(
                    imagefile, "".join(lines[max_lines_per_bubble:])
                )
            )

    elif text != "":
        bubble_x1 = -font_size
        bubble_y1 = 0
        draw.ellipse(
            [
                (bubble_x1, bubble_y1),
                (bubble_x1 + bubble_width, bubble_y1 + bubble_height),
            ],
            outline="black",
            width=5,
            fill="white",
        )
        draw.polygon(
            (
                (bubble_x1 + bubble_width - 7, (bubble_y1 + bubble_height) // 2),
                (
                    bubble_x1 + bubble_width + ((x1 - (bubble_x1 + bubble_width)) // 2),
                    y2,
                ),
                (
                    bubble_x1 + bubble_width - 7,
                    ((bubble_y1 + bubble_height) // 2) - font_size,
                ),
            ),
            fill=(255, 255, 255),
        )
        draw.line(
            (
                (bubble_x1 + bubble_width - 4, (bubble_y1 + bubble_height) // 2),
                (
                    bubble_x1 + bubble_width + ((x1 - (bubble_x1 + bubble_width)) // 2),
                    y2,
                ),
                (
                    bubble_x1 + bubble_width - 5,
                    ((bubble_y1 + bubble_height) // 2) - font_size,
                ),
            ),
            fill=(0, 0, 0),
            width=6,
        )

        if (
            len(lines) > max_lines_per_bubble
            and image.width - x2 > bubble_width - font_size // 2
        ):
            bubble_x2 = bubble_x1
            bubble_y2 = bubble_y1
            bubble_x1 = x2 + font_size // 2
            bubble_y1 = 0
            draw.ellipse(
                [
                    (bubble_x1, bubble_y1),
                    (bubble_x1 + bubble_width, bubble_y1 + bubble_height),
                ],
                outline="black",
                width=5,
                fill="white",
            )
            draw.polygon(
                (
                    (bubble_x1 + 7, (bubble_y1 + bubble_height) // 2),
                    (x2 + ((bubble_x1 - x2) // 2), y2),
                    (
                        bubble_x1 + 7,
                        ((bubble_y1 + bubble_height) // 2) - font_size,
                    ),
                ),
                fill=(255, 255, 255),
            )
            draw.line(
                (
                    (bubble_x1 + 5, (bubble_y1 + bubble_height) // 2),
                    (x2 + ((bubble_x1 - x2) // 2), y2),
                    (
                        bubble_x1 + 5,
                        ((bubble_y1 + bubble_height) // 2) - font_size,
                    ),
                ),
                fill=(0, 0, 0),
                width=6,
            )

        elif len(lines) > max_lines_per_bubble:
            rec = True
            results.append(
                put_speech_bubble_on_image(
                    imagefile, "".join(lines[max_lines_per_bubble:])
                )
            )

    if text != "":
        # 全体の列数を計算
        num_columns = len(lines)
        if num_columns > max_lines_per_bubble:
            num_columns = max_lines_per_bubble

        # # テキスト全体の幅を計算
        # total_text_width = num_columns * (font_size + 10)

        line_spacing = 10

        # # 吹き出し全体の中央にテキストを配置するための初期X座標を計算
        text_x = (
            (bubble_x1 + (bubble_width // 2))
            + (font_size // 2) * (num_columns - 2)
            + (line_spacing // 2) * (num_columns - 1)
        )
        text_y = (
            bubble_y1 + font_size * 2
        )  # 吹き出しの上端から二文字分下の位置にテキストを描画

        # 複数行を描画する
        # 吹き出し当たりの最大行数以内の場合
        if len(lines) <= max_lines_per_bubble:
            for line in lines:
                for char in line:
                    if char == "ー":
                        char = "|"
                    draw.text(
                        (
                            text_x + special_char_offset(char, font_size)[0],
                            text_y + special_char_offset(char, font_size)[1],
                        ),
                        char,
                        fill="black",
                        font=font,
                        align="center",
                    )
                    # 次の文字のY座標を更新
                    # text_bbox = draw.textbbox((0,0), char, font=font)
                    # char_height = text_bbox[1] - text_bbox[3]
                    text_y += font_size + 0

                # 次の行に移るため、X座標をずらす
                text_x -= font_size + line_spacing  # 少し余白を加えて次の行に進む
                text_y = bubble_y1 + font_size * 2  # Y位置をリセットして次の行を開始

        # 吹き出し当たりの最大行数を超える場合
        elif max_lines_per_bubble < len(lines) and not rec:

            for line in lines[0:max_lines_per_bubble]:
                for char in line:
                    if char == "ー":
                        char = "|"
                    draw.text(
                        (
                            text_x + special_char_offset(char, font_size)[0],
                            text_y + special_char_offset(char, font_size)[1],
                        ),
                        char,
                        fill="black",
                        font=font,
                        align="center",
                    )
                    text_y += font_size + 0

                # 次の行に移るため、X座標をずらす
                text_x -= font_size + line_spacing  # 少し余白を加えて次の行に進む
                text_y = bubble_y1 + font_size * 2  # Y位置をリセットして次の行を開始

            if len(lines) > 2 * max_lines_per_bubble:
                num_columns = max_lines_per_bubble
            else:
                num_columns = len(lines) - max_lines_per_bubble

            # 二個目の吹き出し全体の中央にテキストを配置するための初期X座標を再計算
            text_x = (
                (bubble_x2 + (bubble_width // 2))
                + (font_size // 2) * (num_columns - 2)
                + (line_spacing // 2) * (num_columns - 1)
            )
            text_y = (
                bubble_y1 + font_size * 2
            )  # 吹き出しの上端から二文字分下の位置にテキストを描画

            for line in lines[max_lines_per_bubble : 2 * max_lines_per_bubble]:
                for char in line:
                    if char == "ー":
                        char = "|"
                    draw.text(
                        (
                            text_x + special_char_offset(char, font_size)[0],
                            text_y + special_char_offset(char, font_size)[1],
                        ),
                        char,
                        fill="black",
                        font=font,
                        align="center",
                    )
                    text_y += font_size + 0

                # 次の行に移るため、X座標をずらす
                text_x -= font_size + line_spacing  # 少し余白を加えて次の行に進む
                text_y = bubble_y2 + font_size * 2  # Y位置をリセットして次の行を開始

            if len(lines) > max_lines_per_bubble * 2:
                put_speech_bubble_on_image(
                    imagefile, "".join(lines[2 * max_lines_per_bubble :])
                )

        elif rec:
            rec = False
            for line in lines[0:max_lines_per_bubble]:
                for char in line:
                    if char == "ー":
                        char = "|"
                    draw.text(
                        (
                            text_x + special_char_offset(char, font_size)[0],
                            text_y + special_char_offset(char, font_size)[1],
                        ),
                        char,
                        fill="black",
                        font=font,
                        align="center",
                    )
                    text_y += font_size + 0

                # 次の行に移るため、X座標をずらす
                text_x -= font_size + line_spacing  # 少し余白を加えて次の行に進む
                text_y = bubble_y1 + font_size * 2  # Y位置をリセットして次の行を開始

    results.append(image)
    image.save(imagefile.replace(".png", "") + "_fukidashi.png")
    image.show()

    return results


imgarray = []
font_size = 50

# scr = "ああああああああああああああああっ。"
scr = "ひらがなぁあああカタカナッーーー漢字・漢字、漢字。ALPHABETalphabet1234！？"
scr = input("script? > ")
img_num = int(input("imagenum? > ")) or 1


for i in range(1, 10):
    imgarray.append("./image/manga_tanakasiten/img" + str(i) + ".png")

for i in range(0, 9):
    put_speech_bubble_on_image(imgarray[i], scr)
