# RNN-Transducer

## Intro
한국어를 위한 RNN-Transducer입니다.

## Data set
### AI_hub
AI hub에서 제공하는 '한국어 음성데이터'를 사용하였습니다. AI Hub 음성 데이터는 다음 링크에서 신청 후 다운로드 하실 수 있습니다.

AI Hub 한국어 음성 데이터 : http://www.aihub.or.kr/aidata/105 

### Custom Data set
데이터를 커스텀하여 사용하고 싶으신분들은 다음과 같은 형식으로 .csv 파일을 제작하면 됩니다.

 '''
 wav_path/wav_name.wav, txt_path/txt_path.txt
 '''

### Labeling
 ''' 
 #id\char # vocab size = 54
 0   _
 1    
 2   ㄱ
 ...
 52   ㅄ
 53   <s>
  54   </s>
 '''

음절 단위 말고 초성 중성 종성으로 나눈 이유는 RNN-T loss를 사용하게 되면 vocab size가 증가 할 수록 메모리를 많이 잡아 먹기 때문에 학습에서 이득을 보기 위해 다음과 같이 사용하였습니다.
ex) 사과 -> ㅅ ㅏ ㄱ ㅘ

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

## Contacts
해당 작업에 대한 피드백, 문의사항 모두 환영합니다.

fd873630@naver.com로 메일주시면 최대한 빨리 답장드리겠습니다.

인하대학교 전자공학과 4학년 정지호