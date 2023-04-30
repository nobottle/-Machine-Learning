# Linear Regression pytorch

```python
import torch #torch라이브러리 불러옴
import torch.nn as nn #torch.nn모듈에서 nn이라는 별칭으로 import >>신경망 모델의 다양한
#클래스와 함수를 제공
import torch.nn.function as F #신경망 모델의 다양한 함수를 제공
import torch.optim as optim #최적화 알고리즘에 관한거가 있음

#변수선언단계
x_train = torch.FloatTensor([[1], [2], [3]]) #독립변수
y_train = torch.FloatTensor([[2], [4], [6]]) #종속변수

# optimizer 설정
optimizer = optim.SGD([W, b], lr=0.01) #w,b의 최적을 찾는 과정

nb_epochs = 1000 # 원하는만큼 경사 하강법을 반복함
for epoch in range(nb_epochs + 1):

    # H(x) 계산
    hypothesis = x_train * W + b #ax+b같은 형태니까

    # cost 계산
    cost = torch.mean((hypothesis - y_train) ** 2) #linear regression형태

    # cost로 H(x) 개선
    optimizer.zero_grad() #일단 0으로 초기화
    cost.backward() #비용 함수를 미분하여 gradient 계산
    optimizer.step() #W와 b를 업데이트함ㄴ

    # 100번마다 로그 출력
    if epoch % 100 == 0:
        print('Epoch {:4d}/{} W: {:.3f}, b: {:.3f} Cost: {:.6f}'.format(
            epoch, nb_epochs, W.item(), b.item(), cost.item()
        ))
```