using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.MLAgents;

public class RewardManager : MonoBehaviour
{
    // for marl_marl env (because env code is not good)
    public float ShootRewardWeight=0.25f;
    public float KillRewardWeight=1f;
    public float HpRewardWeight=1f;
    public float HpPenaltyWeight=1f;
    public float DefthPenaltyWeight=1f;
    // [access_level] [return_type] [function_name]() {}
    // Agent & ENV Reward Function

    public float ShootingReward(List<MARL_MARLagent> AgentList)
    {
        // Agents shooting reward, depend on Position
        // call when agent shoot enemy
        float agent_shoot = 0f;
        foreach (var agent in AgentList)
        {
            if (agent.StepShootingReward != 0f)
            {
                agent_shoot += agent.StepShootingReward;
                agent.StepShootingReward = 0f;
            }
        }
        return ShootRewardWeight * agent_shoot;
    }

    public float HpReward(List<MARL_MARLagent> AgentList)
    {
        // Agent Team Hp Reward
        float agent_hp = 0f;
        foreach (var agent in AgentList)
        {
            if (agent.hp > 0f) { agent_hp += agent.hp; }
        }
        return HpRewardWeight * agent_hp;
    }

    public float KillReward(List<MARL_MARLagent> EnemyList)
    {
        // Agent Team Kill Reward
        float agent_kill = 0f;
        foreach (var enemy in EnemyList)
        {
            if (enemy.Stepdied) 
            { 
                agent_kill += enemy.rewardWeight; 
            }
        }
        return KillRewardWeight * agent_kill;
    }

    public float HpPenalty(float max_hp, List<MARL_MARLagent> AgentList)
    {
        // Agent Team Hp Reward
        float agent_hp = 0f;
        foreach (var agent in AgentList)
        {
            if (agent.hp > 0f) { agent_hp += agent.hp; }
        }
        float StepHpPenalty = HpPenaltyWeight * (max_hp - agent_hp) / max_hp;
        return StepHpPenalty;
    }

    public float DeathPenalty(float max_hp, List<MARL_MARLagent> AgentList)
    {
        // Agent Team Death Penalty
        float agent_death = 0f;
        foreach (var agent in AgentList)
        {
            if (agent.Stepdied) 
            { 
                agent_death += agent.rewardWeight; 
                agent.Stepdied = false;
            }
        }
        return DefthPenaltyWeight * agent_death;
    }
}
