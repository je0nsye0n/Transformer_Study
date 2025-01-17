import numpy as np
import pandas as pd
import torch
from Korpora import Korpora
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler


# 1. 디바이스 설정
device = "cuda" if torch.cuda.is_available() else "cpu"

# 2. 데이터 로드 및 준비
corpus = Korpora.load("nsmc")
df = pd.DataFrame(corpus.test).sample(20000, random_state=42)

# 데이터셋 분리
test = df.sample(frac=0.2, random_state=42)

# 3. 토크나이저 준비
tokenizer = BertTokenizer.from_pretrained(
    pretrained_model_name_or_path="bert-base-multilingual-cased",
    do_lower_case=False
)

# 데이터셋 생성 함수
def make_dataset(data, tokenizer, device):
    tokenized = tokenizer(
        text=data.text.tolist(),
        padding="longest",
        truncation=True,
        return_tensors="pt"
    )
    input_ids = tokenized["input_ids"].to(device)
    attention_mask = tokenized["attention_mask"].to(device)
    labels = torch.tensor(data.label.values, dtype=torch.long).to(device)
    return TensorDataset(input_ids, attention_mask, labels)

test_dataset = make_dataset(test, tokenizer, device)
test_dataloader = DataLoader(
    test_dataset, sampler=SequentialSampler(test_dataset), batch_size=32
)

# 4. 평가 함수 정의
def evaluation(model, dataloader):
    with torch.no_grad():
        model.eval()
        criterion = torch.nn.CrossEntropyLoss()
        val_loss, val_accuracy = 0.0, 0.0
        
        for input_ids, attention_mask, labels in dataloader:
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            logits = outputs.logits

            loss = criterion(logits, labels)
            logits = logits.detach().cpu().numpy()
            label_ids = labels.to("cpu").numpy()
            accuracy = np.sum(np.argmax(logits, axis=1) == label_ids) / len(label_ids)
            
            val_loss += loss.item()
            val_accuracy += accuracy
    
    val_loss = val_loss / len(dataloader)
    val_accuracy = val_accuracy / len(dataloader)
    return val_loss, val_accuracy

# 5. 모델 로드 및 가중치 적용
model = BertForSequenceClassification.from_pretrained(
    pretrained_model_name_or_path="bert-base-multilingual-cased",
    num_labels=2
).to(device)

# 저장된 가중치 로드
model.load_state_dict(torch.load("./models/BertForSequenceClassification.pt"))

# 모델을 평가 모드로 전환
model.eval()

# 6. 테스트 데이터로 평가
test_loss, test_accuracy = evaluation(model, test_dataloader)
print(f"Test Loss : {test_loss:.4f}")
print(f"Test Accuracy : {test_accuracy:.4f}")


# 1. 테스트 데이터에서 일부 샘플 선택
sample_data = test.sample(5, random_state=42)  # 테스트 데이터에서 5개 샘플 선택
sample_dataset = make_dataset(sample_data, tokenizer, device)

# 2. 예측 수행
model.eval()
predictions = []
labels = []

with torch.no_grad():
    for input_ids, attention_mask, label in DataLoader(sample_dataset, batch_size=1):
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        pred_label = torch.argmax(logits, axis=1).item()  # 예측된 레이블
        predictions.append(pred_label)
        labels.append(label.item())

# 3. 예측 결과 출력
for i in range(len(sample_data)):
    print(f"텍스트: {sample_data.text.iloc[i]}")
    print(f"실제 레이블: {labels[i]} (0: 부정, 1: 긍정)")
    print(f"예측 레이블: {predictions[i]} (0: 부정, 1: 긍정)")
    print("-" * 50)
