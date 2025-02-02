# 트랜스포머 모듈 구현 내용 정리
* 인코더 : Input  → Positional Encoding  → <span style="color:red"> MHA module </span>  → Add&Norm  →  <span style="color:red"> FFN module </span> → Add&Norm
* 주요 모듈인 Attn과 FFN에 대한 C 구현 방식은 아래와 같다
* C로 정확하게 구현했는지 확인하기 위해 필요한 모든 데이터의 값(Query, Key, Value, weight, bias)은 Python에서 사용한 값을 저장하여 C에서 로드해오도록 구현하였다. (각각 data와 data2 폴더에 필요한 값들을 쓰고 읽는다.)
---
### Multi-Head Attention module
이 코드는 Multi-Head Attention (MHA) 를 구현한 것으로, 주어진 Query, Key, Value를 입력받아 어텐션 연산을 수행한 후 최종 출력을 반환한다. 핵심 로직은 다음과 같다:

1. 데이터 로드 및 초기화
* Query, Key, Value 및 가중치 로드
* 동적 메모리 할당을 통해 데이터 저장
   
2. 선형 변환 및 헤드 분리
* Query, Key, Value에 대해 개별적으로 선형 변환 수행 (LinearMapping)
* d_k 단위로 분할하여 다중 헤드 형태로 변환

3. 어텐션 스코어 계산
* Attention_h() 함수에서 Query와 Key 내적 후 Softmax 적용
* Attention Value 계산하여 최종 출력

4. 결과 결합 및 최종 선형 변환
* 헤드별 출력을 Concatenation 후 최종 선형 변환 수행

### Feedforward module
이 코드는 Feedforward Neural Network (FFN) 을 구현한 것으로, Transformer 블록 내의 Feedforward Layer를 수행하는 역할을 한다. 주요 로직은 다음과 같다:

1. 데이터 로드 및 초기화
* 입력 데이터 (input)을 파일에서 읽어옴
* 가중치 및 편향 (weights, biases)을 동적 메모리 할당 후 파일에서 로드

2. Feedforward Layer 수행
* Feedforward 확장 (feedforward1()): 입력 차원(D_MODEL)을 확장하여 (HIDDEN) 변환 후 ReLU 적용
* Feedforward 축소 (feedforward2()): 확장된 차원을 다시 축소하여 (D_MODEL) 최종 출력을 생성

3. 출력 출력 및 메모리 해제
* 최종 출력을 print_output()을 통해 출력
* 동적으로 할당한 메모리를 반환


