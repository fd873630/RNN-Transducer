# RNN-Transducer

## Intro
한국어를 위한 RNN-Transducer입니다. 실시간 인식에는 attention기반의 모델보다 RNN-Transducer가 효과적이다고 합니다. 현재 git hub에는 한국어로 test한 결과가 없어 한국어 RNN-Transducer의 성능을 확인하기 위해 작성하였습니다.

## Version
* torch version = 1.2.0
* Cuda compilation tools, release 9.1, V9.1.85

## Model
<img width = "400" src = "https://user-images.githubusercontent.com/43025347/96749425-c6f0e480-1405-11eb-9328-06010a44f839.png">

## Data set
### AI_hub
AI hub에서 제공하는 '한국어 음성데이터'를 사용하였습니다. AI Hub 음성 데이터는 다음 링크에서 신청 후 다운로드 하실 수 있습니다.

train data 총 길이 - 약 250시간 (248.9시간) "./label,csv/AI_hub_train_U_800_T_50.csv"
val data 총 길이 - 약 5시간 (5.1시간) "./label,csv/AI_hub_val_U_800_T_50.csv"

AI Hub 한국어 음성 데이터 : http://www.aihub.or.kr/aidata/105 

Q1 : 왜 AI hub데이터에 있는 eval 데이터 셋을 사용하지 않고 train에서 임의로 나눠 사용했는지?
A1 : RNN-T loss는 wav len과 script len에 따라서 시간과 메모리를 잡아 먹습니다. 그러므로 wav len과 script len은 특정 길이로 제한했는데 eval 데이터에서 제한하면 데이터가 부족해 train에서 나눴습니다. (옛날(19년)에는 없었는데 최근에 올라온거라 ...)

### Custom Data set
데이터를 커스텀하여 사용하고 싶으신분들은 다음과 같은 형식으로 .csv 파일을 제작하면 됩니다.

    wav_path/wav_name.wav, txt_path/txt_path.txt

### Script
* Raw script
b/ (70%)/(칠 십 퍼센트) 확률이라니 

* Final script
ㅊㅣㄹ ㅅㅣㅂ ㅍㅓㅅㅔㄴㅌㅡ ㅎㅘㄱㄹㅠㄹㅇㅣㄹㅏㄴㅣ

위의 txt 전처리는 https://github.com/sooftware/KoSpeech/wiki/Preparation-before-Training 다음을 참고하였습니다.

### Labeling

    #id\char 
    0   _
    1    
    2   ㄱ
    ...
    52   ㅄ
    53   <s>
    54   <//s>
 
음절 단위 말고 자소 단위로 나눈 이유는 RNN-T loss는 wav len과 script len뿐만 아니라 vocab size도 메모리를 잡아 먹습니다.즉 vocab size가 증가 할 수록 메모리를 많이 잡아 먹기 때문에 학습에서 gpu 메모리 이득을 보기 위해 다음과 같이 사용하였습니다. (gpu 메모리가 여유가 있으시면 음절 단위로 해보셔도 좋을것 같습니다.)

* Final script
ㅊㅣㄹ ㅅㅣㅂ ㅍㅓㅅㅔㄴㅌㅡ ㅎㅘㄱㄹㅠㄹㅇㅣㄹㅏㄴㅣ

* Number Final script
16 41 7 1 11 41 9 1 19 25 11 26 4 18 39 ...

txt_path.txt 에 Number Final script가 들어가야 합니다.

Q1 : <s> 와 </s> 는 왜 들어간건지? 
A1 : 나중에 two pass를 사용하기 위해서 집어 넣었습니다. RNN-T만 사용하신다면 삭제해도 무방합니다.


## References
### Git hub References
* https://github.com/1ytic/warp-rnnt
* https://github.com/ZhengkunTian/rnn-transducer
* https://github.com/HawkAaron/E2E-ASR
* https://github.com/sooftware/KoSpeech/wiki/Preparation-before-Training

### Paper References
* https://arxiv.org/abs/1211.3711

### Blog References
* https://gigglehd.com/zbxe/14052329
* https://dos-tacos.github.io/paper%20review/sequence-transduction-with-rnn/

### Youtube References
* https://www.youtube.com/watch?v=W7b77hv3Rak&ab_channel=KrishnaDN

## computer power
* NVIDIA TITAN Xp * 4

## Contacts
학부생의 귀여운 시도로 봐주시고 해당 작업에 대한 피드백, 문의사항 모두 환영합니다.

fd873630@naver.com로 메일주시면 최대한 빨리 답장드리겠습니다.

인하대학교 전자공학과 4학년 정지호