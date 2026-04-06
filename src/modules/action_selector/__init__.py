from .action_selector import single_actor_selector, multi_actor_selector

action_selector_registry = {}

action_selector_registry["irf"] = single_actor_selector
action_selector_registry["coma"] = single_actor_selector
action_selector_registry["cds"] = single_actor_selector
action_selector_registry["emc"] = single_actor_selector
action_selector_registry["poca"] = single_actor_selector