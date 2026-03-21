# 아나콘다 환경 ML-Agent Setting 설명서입니다.

사용자가 바꿀 수 있는 부분은 괄호 안에 표시했습니다. 
Ex : (내가 원하는 가상환경 이름) 

### 1. 아나콘다를 설치
https://github.com/Unity-Technologies/ml-agents/blob/develop/docs/Installation-Anaconda-Windows.md  
아나콘다 설치는 해당 문서 참고  
  
### 2. 아나콘다 실행한 뒤 가상 환경 생성
<pre>
<code>conda create -n (내가 원하는 가상환경 이름) python=(3.10.12 :공식에서 권장하는 버전)</code>
</pre>
  
중간에 yes/no 선택하는거 뜨는데 yes 누르면 진행됩니다.

### 3. 가상환경 활성화
<pre>
<code>conda activate (내가 설정한 가상환경 이름)</code>
</pre>  

### 4. Github ML-Agent 패키지 다운로드
#### 4.1 공식 권장 사항 - git이 세팅되어 있다면 이렇게 해주면 됩니다.
  
    git clone --branch release_21 https://github.com/Unity-Technologies/ml-agents.git
  
#### 4.2. git 없으면 그냥 mlagent release 21 zip파일 받으신 뒤 압축 해제 하세요
  (그냥 본인이 사용하고싶은 버전 다운 받으셔도 됩니다. release 20이나 22같은거)

### 5. Pytorch 설치
공식 사이트에 나와있는 command는 다음과 같습니다만...
<pre>
<code>pip3 install torch~=2.2.1 --index-url https://download.pytorch.org/whl/cu121</code>
</pre>

만약 Cuda 사용하신다면 Pytorch도 mlagent 버전 + 본인 Cuda 버전에 맞게 까셔야합니다.  
https://pytorch.org/get-started/previous-versions/  
여기서 복사해서 커맨드창에 그대로 붙여넣기 하신 뒤 엔터 누르면 됩니다.  

### 6. mlagents 파이썬 패키지 설치
<pre>
<code>cd (본인 mlagents 폴더 경로)</code>
</pre>
먼저 위와 같이 mlagents가 설치되어있는 폴더로 이동합니다.
<pre>
<code>cd ml-agents-envs  
pip install -e .
cd ..
cd ml-agents
pip install -e .</code>
</pre>
복붙하고 엔터 누르면 끝납니다.
