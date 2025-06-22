from text_based_split.id_match_rigid import predict as id_match_rigid_predict
from text_based_split.id_match import predict as id_match_predict
from text_based_split.large_texts import predict as large_texts_predict
from text_based_split.page_number import predict as page_number_predict
from text_based_split.web_match import predict as web_match_predict


def predict(pdf_path: str):
    id_rigid_pred = id_match_rigid_predict(pdf_path)
    id_pred = id_match_predict(pdf_path)
    large_texts_pred = large_texts_predict(pdf_path)
    page_number_pred = page_number_predict(pdf_path)
    web_match_pred = web_match_predict(pdf_path)
    n = len(id_rigid_pred)
    pred = [1] * n
    for i in range(n):
        if id_rigid_pred[i] == 0:
            pred[i] = 0
            continue
        if large_texts_pred[i] == 0:
            pred[i] = 0
            continue
        if page_number_pred[i] == 0:
            pred[i] = 0
            continue
        if web_match_pred[i] == 0:
            pred[i] = 0
            continue
        if id_pred[i] == 0:
            pred[i] = 0
            continue
    return pred
