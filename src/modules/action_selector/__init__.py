from .action_selector import single_actor_selector, multi_actor_selector
from .upper_actor import upper_action_selector

# 알고리즘별 액션 선택 함수 레지스트리
# 에이전트의 행동 선택 신경망인 'actor' 개수에 따라 액션 선택 함수 레지스트리 구성
action_selector_registry = {}

action_selector_registry["liir"] = single_actor_selector
action_selector_registry["coma"] = single_actor_selector
action_selector_registry["cds"] = single_actor_selector
action_selector_registry["emc"] = single_actor_selector
action_selector_registry["poca"] = single_actor_selector

# 상위 에이전트의 행동 선택 신경망인 ppo는 upper_action_selector 사용용
action_selector_registry["ppo"] = upper_action_selector
