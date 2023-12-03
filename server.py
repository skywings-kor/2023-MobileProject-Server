from flask import Flask, request, jsonify
from transformers import BertConfig, BertForSequenceClassification, BertTokenizer
import torch
import json
import openai
from fuzzywuzzy import fuzz


# Flask 앱 초기화
app = Flask(__name__)

openai.api_key = ''

# 질문-대답 데이터 로드
with open('./qa_data.json', 'r', encoding='utf-8') as f:
    qa_pairs = json.load(f)



# 모델 및 토크나이저 로드
args = ClassificationDeployArguments(
    pretrained_model_name="beomi/kcbert-base",
    downstream_model_dir="./model",
    max_seq_length=128,
)

fine_tuned_model_ckpt = torch.load(
    args.downstream_model_checkpoint_fpath,
    map_location=torch.device("cpu")
)

pretrained_model_config = BertConfig.from_pretrained(
    args.pretrained_model_name,
    num_labels=fine_tuned_model_ckpt['state_dict']['model.classifier.bias'].shape.numel(),
)

model = BertForSequenceClassification(pretrained_model_config)
model.load_state_dict({k.replace("model.", ""): v for k, v in fine_tuned_model_ckpt['state_dict'].items()})
model.eval()

tokenizer = BertTokenizer.from_pretrained(
    args.pretrained_model_name,
    do_lower_case=False,
)

# 추론 함수
def inference(sentence):
    inputs = tokenizer(
        [sentence],
        max_length=args.max_seq_length,
        padding="max_length",
        truncation=True,
    )
    with torch.no_grad():
        outputs = model(**{k: torch.tensor(v) for k, v in inputs.items()})
        prob = outputs.logits.softmax(dim=1)
        positive_prob = round(prob[0][1].item(), 4)
        negative_prob = round(prob[0][0].item(), 4)
        pred = "긍정 (positive)" if torch.argmax(prob) == 1 else "부정 (negative)"
    return {
        'sentence': sentence,
        'prediction': pred,
        'positive_prob': positive_prob,
        'negative_prob': negative_prob,
    }

# API 엔드포인트
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    sentence = data['sentence']
    result = inference(sentence)
    return jsonify(result)



# 챗봇 라우트 설정
@app.route('/chatbot', methods=['POST'])
def chatbot():
    user_input = request.json.get('question', '')

    best_match = None
    best_score = -1

    for qa_pair in qa_pairs:
        similarity_score = fuzz.token_set_ratio(user_input, qa_pair["question"])
        if similarity_score > best_score:
            best_score = similarity_score
            best_match = qa_pair

    if best_score >= 70:
        response = best_match['answer']
    elif best_score < 20:
        response = "해당 질문에 대한 답변을 드리기 어렵습니다. 고객센터로 문의 부탁드립니다."
    else:
        response = "질문을 더 자세히 해주십시오."

    return jsonify({"response": response})



# 앱 실행
if __name__ == "__main__":
    app.run(host='0.0.0.0', port='5000', debug=True)