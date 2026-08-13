from bench.score import parse_bbox, score


def test_scorers() -> None:
    assert score({"scorer": "contains", "expected": {"text": "EXIT 42"}}, "The sign says EXIT 42.")["pass"]
    assert not score({"scorer": "contains", "expected": {"text": "EXIT 42"}}, "hello")["pass"]
    assert score({"scorer": "contains", "expected": {"text": "red"}}, "It is Red.")["pass"]
    assert not score({"scorer": "contains", "expected": {"text": "red"}}, "The box is colored blue")["pass"]
    assert not score({"scorer": "contains", "expected": {"text": "no"}}, "I don't know")["pass"]
    assert score({"scorer": "contains", "expected": {"text": "5"}}, "5")["pass"]
    assert not score({"scorer": "contains", "expected": {"text": "5"}}, "15")["pass"]
    assert score({"scorer": "contains", "expected": {"text": "18.50"}}, "TOTAL $18.50")["pass"]
    assert parse_bbox('[{"label": "x", "bbox": [0.10, 0.20, 0.80, 0.90]}]') == [100, 200, 800, 900]
    assert parse_bbox("[100, 200, 800, 900]") == [100, 200, 800, 900]
    assert score(
        {"scorer": "bbox_iou", "expected": {"bbox": [100, 100, 200, 200], "iou": 0.5}},
        "here [100, 100, 200, 200]",
    )["pass"]
    assert score(
        {"scorer": "click_in_box", "expected": {"bbox": [0, 0, 500, 500]}},
        "[100, 100, 200, 200]",
    )["pass"]
    assert not score(
        {"scorer": "click_in_box", "expected": {"bbox": [0, 0, 50, 50]}},
        "[400, 400, 500, 500]",
    )["pass"]
    assert score(
        {"scorer": "click_in_box", "expected": {"bbox": [0, 0, 500, 500]}},
        "pyautogui.click(x=100, y=200)",
    )["pass"]
    assert score(
        {"scorer": "click_in_box", "expected": {"bbox": [0, 0, 500, 500]}},
        "[0.1, 0.2]",
    )["pass"]
    assert score(
        {"scorer": "tool_call", "expected": {"name": "get_weather", "arguments": {"city": "Paris"}}},
        '{"name": "get_weather", "arguments": {"city": "Paris"}}',
    )["pass"]
    assert score(
        {"scorer": "tool_call", "expected": {"name": "get_weather", "arguments": {"city": "Paris"}}},
        '<|tool_call_start|>[get_weather(city="Paris")]<|tool_call_end|>',
    )["pass"]
    assert score({"scorer": "if_words", "expected": {"n": 3}}, "one two three")["pass"]
    assert score({"scorer": "if_json", "expected": {"object": {"ok": True, "n": 2}}}, '{"ok": true, "n": 2}')["pass"]
    assert score({"scorer": "if_forbidden", "expected": {"letter": "e"}}, "a cat sat")["pass"]
    layout_task = {
        "scorer": "layout",
        "expected": {
            "iou": 0.3,
            "min_recall": 0.5,
            "regions": [
                {"label": "page_header", "bbox": [37, 31, 250, 54]},
                {"label": "title", "bbox": [78, 148, 486, 190]},
                {"label": "table", "bbox": [78, 437, 921, 656]},
                {"label": "page_footer", "bbox": [78, 912, 197, 942]},
            ],
        },
    }
    layout_out = (
        "image_index=0 page_header [34, 28, 252, 60]\nACME Docs\n\n"
        "image_index=0 title [75, 145, 491, 183]\nQuarterly Report\n\n"
        "image_index=0 table [76, 436, 926, 658]\n\n"
        "image_index=0 page_number [75, 911, 194, 943]\npage 1\n"
    )
    assert score(layout_task, layout_out)["pass"]
    print("scorers ok")


if __name__ == "__main__":
    test_scorers()
