import argparse

def parse_args():
    """
    명령줄 인자를 파싱하는 함수입니다.
    
    Returns:
        argparse.Namespace: 파싱된 명령줄 인자가 포함된 객체
    """
    parser = argparse.ArgumentParser(description='Run Unity ML-Agents with custom configurations.')
    parser.add_argument('--workerid', type=int, default=0, help='Worker ID for Unity Environment (default: 0)')
    parser.add_argument('--graphic', type=str, default="False", help='no-grphic option for Unity Environment (default: false)')
    # parser.add_argument('--actor', type=str, default='rnn', help='Configuration for team0 (e.g., rnn)')
    
    return parser.parse_args() 