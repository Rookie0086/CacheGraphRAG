import re


def checkanswer(prediction, ground_truth, verbose=False):
    """
    检查预测答案是否与标准答案匹配。

    :param str prediction:
        预测答案，输入字符串将被转换为小写以进行比较。

    :param ground_truth:
        默认为列表，如果输入为str，将手动转为列表，其中列表中的元素表示为候选答案。
        如果是嵌套列表表示这个问题同时包括多个答案，需要同时回答正确。

    :return:
        二进制标签列表，1 表示匹配成功，0 表示匹配失败。
    :rtype: List[int]

    :示例:

    >>> prediction = "The cat sits on the mat"
    >>> ground_truth = [["cat", "CAT"]]
    >>> checkanswer("cat", ground_truth)
    [1]

    >>> checkanswer("cat and mat", [["cat"], ["MAT", "mat"]])
    [1, 1]
    """
    def _normalize_answer(text: str) -> str:
        normalized = text.lower().strip()
        normalized = re.sub(r"[\"'`]", "", normalized)
        normalized = normalized.replace("-", " ")
        normalized = re.sub(r"[\.,，。;；:：]", " ", normalized)
        normalized = normalized.rstrip("!?！？")
        normalized = " ".join(normalized.split())
        return normalized

    prediction = _normalize_answer(prediction)
    if not isinstance(ground_truth, list):
        ground_truth = [ground_truth]
    labels = []
    for instance in ground_truth:
        flag = True
        if isinstance(instance, list):
            flag = False
            instance = [_normalize_answer(i) for i in instance]
            for i in instance:
                if i in prediction:
                    flag = True
                    break
        else:
            instance = _normalize_answer(instance)
            if instance not in prediction:
                flag = False
        labels.append(int(flag))

    if verbose:
        print_text(
            f"\nprediction: {prediction}, \nground_truth: {ground_truth}, \nlabels: {labels}\n",
            color="yellow",
        )

    return labels


def get_accuracy(labels, info=None):
    tt = 0
    for label in labels:
        if 0 not in label and 1 in label:
            tt += 1
    acc = tt / len(labels)

    if info:
        print_text(f"{info} accuracy {acc}\n", color="green")

    return acc
