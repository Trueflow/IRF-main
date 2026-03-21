using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.MLAgents.Policies;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using JetBrains.Annotations;
using static HRL_MARL_controller;
using System.Linq;
using System;

public class marl_marl_Controller : MonoBehaviour
{
    // MARL vs RSA Environment Controller
    [Tooltip("Max Environment Steps")] public int MaxEnvironmentSteps;
    private int ResetTimer = 0;
    public GameObject floor;
    public GameObject BlueArea;
    public GameObject RedArea;
    // public List<GameObject> AreaList = new();
    [HideInInspector] public Vector3 BlueAreaPos;
    [HideInInspector] public Vector3 RedAreaPos;
    [HideInInspector] bool ResetComplete = false;

    [HideInInspector] public SimpleMultiAgentGroup BlueAgentGroup = new SimpleMultiAgentGroup();
    [HideInInspector] public SimpleMultiAgentGroup RedAgentGroup = new SimpleMultiAgentGroup();

    public ObjectCount count;
    public RewardManager rewardManager;
    public bool SparseReward = false;
    // public float DefaultReward = 1f;
    public float hpPenalty = 1f;
    public float DeathPenalty = 1f;
    public float ShootRewardWeight = 0.25f;
    public float WinReward = 3f;
    public float neg_scale = 1f;
    [HideInInspector] public float BlueStepReward = 0f;
    [HideInInspector] public float RedStepReward = 0f;
    [HideInInspector] public float blue_max;
    [HideInInspector] public float red_max;
    [HideInInspector] public float previousBluehp;
    [HideInInspector] public float previousRedhp;

    void Start() // 최초 실행 시 호출됨. 환경에 대한 기본 설정
    {
        int BlueIndex = 0;
        foreach (var agent in count.BlueList)
        {
            BlueAgentGroup.RegisterAgent(agent);
            agent.MyIndex = BlueIndex;
            agent.MyTeam = agent.gameObject.GetComponent<BehaviorParameters>().TeamId;
            BlueIndex++;
            blue_max += agent.hpMax;
            agent.EnemyList = count.RedList;
        }
        int RedIndex = 0;
        foreach (var agent in count.RedList)
        {
            RedAgentGroup.RegisterAgent(agent);
            agent.MyIndex = RedIndex;
            agent.MyTeam = agent.gameObject.GetComponent<BehaviorParameters>().TeamId;
            RedIndex++;
            red_max += agent.hpMax;
            agent.EnemyList = count.BlueList;
        }
        BlueAreaPos = new Vector3(BlueArea.transform.position.x, 0f, BlueArea.transform.position.z);
        RedAreaPos = new Vector3(RedArea.transform.position.x, 0f, RedArea.transform.position.z);
        ResetScene();
        Debug.Log("Env Start");
    }

    public void ShootReward()
    {
        // only shooting reward & hp penalty & death penalty
        float BlueShootingReward = rewardManager.ShootingReward(count.RedList);
        float RedShootingReward = rewardManager.ShootingReward(count.BlueList);
        float BlueHpPenalty = rewardManager.HpPenalty(blue_max, count.BlueList);
        float RedHpPenalty = rewardManager.HpPenalty(red_max, count.RedList);
        float BlueDeathPenalty = rewardManager.DeathPenalty(blue_max, count.BlueList);
        float RedDeathPenalty = rewardManager.DeathPenalty(red_max, count.RedList);

        BlueStepReward = BlueShootingReward - (BlueDeathPenalty + BlueHpPenalty);
        RedStepReward = RedShootingReward - (RedDeathPenalty + RedHpPenalty);
    }

    public void HpReward()
    {
        float BlueHpReward = rewardManager.HpReward(count.BlueList);
        float RedHpReward = rewardManager.HpReward(count.RedList);
        float BlueHpPenalty = rewardManager.HpPenalty(blue_max, count.BlueList);
        float RedHpPenalty = rewardManager.HpPenalty(red_max, count.RedList);
        float BlueDeathPenalty = rewardManager.DeathPenalty(blue_max, count.BlueList);
        float RedDeathPenalty = rewardManager.DeathPenalty(red_max, count.RedList);

        BlueStepReward = BlueHpReward - (BlueDeathPenalty + BlueHpPenalty);
        RedStepReward = RedHpReward - (RedDeathPenalty + RedHpPenalty);
    }
    public void CalculateHpReward()
    {
        float blue_hp = 0f;
        float red_hp = 0f;
        foreach (var blue in count.BlueList)
        {
            if (blue.hp > 0f) { blue_hp += blue.hp; }
        }
        foreach (var red in count.RedList)
        {
            if (red.hp > 0f) { red_hp += red.hp; }
        }
        float red_damaged_ratio = (previousRedhp - red_hp) / red_max; // rsa가 해당 step에서 잃은 체력 비율
        float blue_damaged_ratio = (previousBluehp - blue_hp) / blue_max; // agent가 해당 step에서 잃은 체력 비율
        BlueStepReward = red_damaged_ratio - (blue_damaged_ratio * 0.5f); // 적을 죽이는 방향으로 유도하기 위해
        RedStepReward = blue_damaged_ratio - (red_damaged_ratio * 0.5f);
        previousRedhp = red_hp;
        previousBluehp = blue_hp;
    }

    public void CalculateShootReward() // 보병은 * 1, 기갑은 * 5
    {
        BlueStepReward = 0f;
        RedStepReward = 0f;
        float blue_shoot = 0f;
        float blue_hp = 0f;
        float red_shoot = 0f;
        float red_hp = 0f;
        float blue_death_value = 0f;
        float red_death_value = 0f;
        foreach (var blue in count.BlueList)
        {
            if (blue.StepShootingReward != 0f)
            {
                blue_shoot += blue.StepShootingReward;
                blue.StepShootingReward = 0f;
            }
            // 사격한 적에 따라 가중치를 부여
            if (blue.hp > 0f) { blue_hp += blue.hp; }
            if (blue.Stepdied)
            {
                blue_death_value += blue.rewardWeight;
                blue.Stepdied = false;
            }
        }
        foreach (var red in count.RedList)
        {
            if (red.StepShootingReward != 0f)
            {
                red_shoot += red.StepShootingReward;
                red.StepShootingReward = 0f;
            }
            // 사격한 적에 따라 가중치를 부여
            if (red.hp > 0f) { red_hp += red.hp; }
            if (red.Stepdied)
            {
                red_death_value += red.rewardWeight;
                red.Stepdied = false;
            }
        }
        float blue_damaged_ratio = hpPenalty * (previousBluehp - blue_hp) / blue_max;
        float red_damaged_ratio = hpPenalty * (previousRedhp - red_hp) / red_max;
        BlueStepReward = blue_shoot * ShootRewardWeight - neg_scale * (blue_damaged_ratio + blue_death_value * DeathPenalty);
        RedStepReward = red_shoot * ShootRewardWeight - neg_scale * (red_damaged_ratio + red_death_value * DeathPenalty);
        previousBluehp = blue_hp;
        previousRedhp = red_hp;
    }
    public void FixedUpdate() // 에피소드 종료 조건을 충족하는지 매 step 마다 확인
    {
        if (ResetComplete)
        {
            if (!SparseReward) {
                CalculateShootReward();
                BlueAgentGroup.AddGroupReward(BlueStepReward);
                RedAgentGroup.AddGroupReward(RedStepReward);
            }
            switch (true)
            {
                case true when (ResetTimer >= MaxEnvironmentSteps && MaxEnvironmentSteps > 0):
                    CalculateHP();

                    Debug.Log("Episode Interrupted: Blue " + count.CurrentBlue + " vs Red " + count.CurrentRed);
                    var result = previousBluehp - previousRedhp; // 승리 기준 : 더 많은 체력이 남은 팀이 승리
                    // CalculateReward();
                    if (result < 0) // Red win
                    {
                        count.RedWin = 1;
                        count.BlueWin = -1;
                        if (SparseReward) {
                            BlueAgentGroup.AddGroupReward(-1f);
                            RedAgentGroup.AddGroupReward(1f);
                        }
                        else {
                            BlueAgentGroup.AddGroupReward(-WinReward * neg_scale);
                            RedAgentGroup.AddGroupReward(WinReward);
                        }
                    }
                    else if (result > 0) // Blue win
                    {
                        count.BlueWin = 1;
                        count.RedWin = -1;
                        if (SparseReward) {
                            BlueAgentGroup.AddGroupReward(1f);
                            RedAgentGroup.AddGroupReward(-1f);
                        }
                        else {
                            BlueAgentGroup.AddGroupReward(WinReward);
                            RedAgentGroup.AddGroupReward(-WinReward * neg_scale);
                        }
                    }
                    ResetScene();
                    break;
                case true when (count.CurrentRed == 0 && count.CurrentBlue == 0):
                    Debug.Log("Draw");
                    BlueAgentGroup.EndGroupEpisode();
                    RedAgentGroup.EndGroupEpisode();
                    ResetScene();
                    break;
                case true when (count.CurrentRed == 0):
                    Debug.Log("Agent Blue Win: Blue " + count.CurrentBlue + " vs Red " + count.CurrentRed);
                    // AgentGroup.AddGroupReward(DefaultReward);
                    count.BlueWin = 1;
                    count.RedWin = -1;
                    if (SparseReward) {
                        BlueAgentGroup.AddGroupReward(1f);
                        RedAgentGroup.AddGroupReward(-1f);
                    }
                    else {
                        RedAgentGroup.AddGroupReward(-WinReward * neg_scale);
                        BlueAgentGroup.AddGroupReward(WinReward);
                    }
                    BlueAgentGroup.EndGroupEpisode();
                    RedAgentGroup.EndGroupEpisode();
                    ResetScene();
                    // GC.Collect();
                    // GC.WaitForPendingFinalizers();
                    break;
                case true when (count.CurrentBlue == 0):
                    Debug.Log("Agent Red win: Blue " + count.CurrentBlue + " vs Red " + count.CurrentRed);
                    // AgentGroup.AddGroupReward(-DefaultReward);
                    count.RedWin = 1;
                    count.BlueWin = -1;
                    if (SparseReward) {
                        BlueAgentGroup.AddGroupReward(-1f);
                        RedAgentGroup.AddGroupReward(1f);
                    }
                    else {
                        RedAgentGroup.AddGroupReward(WinReward);
                        BlueAgentGroup.AddGroupReward(-WinReward * neg_scale);
                    }
                    BlueAgentGroup.EndGroupEpisode();
                    RedAgentGroup.EndGroupEpisode();
                    ResetScene();
                    // GC.Collect();
                    // GC.WaitForPendingFinalizers();
                    break;
            }
            ResetTimer++;
        }
    }

    public void CalculateHP()
    {
        float blue_hp = 0f;
        float red_hp = 0f;
        foreach (var blue in count.BlueList)
        {
            if (blue.hp > 0f) { blue_hp += blue.hp; }
        }
        foreach (var red in count.RedList)
        {
            if (red.hp > 0f) { red_hp += red.hp; }
        }
        previousBluehp = blue_hp;
        previousRedhp = red_hp;
    }

    public void ResetScene() // 에피소드가 리셋 될때마다 호출. 위치 등 초기화 해주는 부분
    {
        ResetComplete = false;
        BlueStepReward = 0f;
        RedStepReward = 0f;
        ResetTimer = 0;
        count.BlueWin = 0;
        count.RedWin = 0;
        count.CurrentBlue = count.BlueCount;
        count.CurrentRed = count.RedCount;
        foreach (var agent in count.BlueList)
        {
            ResetAgent(agent, "BlueTeam", BlueArea.transform, BlueAreaPos);
        }
        foreach (var agent in count.RedList)
        {
            ResetAgent(agent, "RedTeam", RedArea.transform, RedAreaPos);
        }
        foreach (var enemy in count.BlueList) { enemy.IsActive = true; }
        foreach (var agent in count.RedList) { agent.IsActive = true; }
        previousBluehp = blue_max;
        previousRedhp = red_max;
        ResetComplete = true;
    }

    public void ResetAgent(MARL_MARLagent agent, string tag, Transform area, Vector3 area_position)
    {
        Rigidbody rigidbody = agent.gameObject.GetComponent<Rigidbody>();
        rigidbody.mass = 5.0f;
        rigidbody.velocity = Vector3.zero; // 속도 초기화
        rigidbody.angularVelocity = Vector3.zero; // 회전 속도 초기화

        float scaleX = area.localScale.x;
        float scaleZ = area.localScale.z;

        float RandomValueX = UnityEngine.Random.Range(-scaleX / 2f, scaleX / 2f);
        float RandomValueZ = UnityEngine.Random.Range(-scaleZ / 2f, scaleZ / 2f);
        agent.transform.SetPositionAndRotation(area_position + new Vector3(RandomValueX, agent.InitialY, RandomValueZ), Quaternion.Euler(0, area.eulerAngles.y, 0));
        Physics.IgnoreCollision(agent.gameObject.GetComponent<Collider>(), floor.GetComponent<Collider>(), false);

        Renderer renderer = agent.gameObject.GetComponent<Renderer>();
        if (renderer != null) { renderer.material = agent.DefaultMaterial; }
        agent.gameObject.tag = tag;
        agent.hp = agent.hpMax;
        agent.remainShootingCool = 0;
        agent.ForwardStick.GetComponent<forwardStickVIsual>().AgentDefault();
    }
}
