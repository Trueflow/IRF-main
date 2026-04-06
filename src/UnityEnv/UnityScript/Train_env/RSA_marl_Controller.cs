using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.MLAgents.Policies;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using JetBrains.Annotations;
using System;
using System.Linq;
using System.Linq.Expressions;

public class RSA_marl_Controller : MonoBehaviour
{
    // MARL vs RSA Environment Controller
    [Tooltip("Max Environment Steps")] public int MaxEnvironmentSteps;
    private int ResetTimer = 0;
    public bool SparseReward = false;
    public GameObject floor;
    public GameObject BlueArea;
    public GameObject RedArea;
    // public List<GameObject> AreaList = new(); // AreaList는 사용되지 않으므로 주석 처리 유지 또는 삭제 고려
    [HideInInspector] public Vector3 BlueAreaPos; // BlueAreaTransform 대신 사용
    [HideInInspector] public Vector3 RedAreaPos;   // RedAreaTransform 대신 사용
    // BlueAreaRotation, RedAreaRotation 필드 제거 (Transform에서 직접 읽어옴)
    [HideInInspector] bool ResetComplete = false;

    [HideInInspector] public SimpleMultiAgentGroup AgentGroup = new SimpleMultiAgentGroup();

    public ObjectCount count;
    public float killRewardWeight = 5f;
    public float hpPenalty = 1f;
    public float DeathPenalty = 1f;
    public float ShootRewardWeight = 0.25f;
    public float WinReward = 3f;
    public float neg_scale = 1f;
    [HideInInspector] public float StepReward = 0f;
    [HideInInspector] public float agent_max;
    [HideInInspector] public float rsa_max;
    [HideInInspector] public float previousAgenthp;
    [HideInInspector] public float previousRSAhp;
    [HideInInspector] public float EpisodeReward = 0f; // 에피소드 종료 시 부여되는 보상

    void Start() // 최초 실행 시 호출됨. 환경에 대한 기본 설정
    {
        int AgentIndex = 0;
        foreach (var agent in count.AgentList)
        {
            AgentGroup.RegisterAgent(agent);
            agent.MyIndex = AgentIndex;
            AgentIndex++;
            agent_max += agent.hpMax;
        }
        int EnemyIndex = 0;
        foreach (var enemy in count.EnemyList)
        {
            enemy.MyIndex = EnemyIndex;
            EnemyIndex++;
            rsa_max += enemy.hpMax;
        }
        // BlueAreaTransform, RedAreaTransform, BlueAreaRotation, RedAreaRotation 초기화 로직 변경
        BlueAreaPos = new Vector3(BlueArea.transform.position.x, 0f, BlueArea.transform.position.z);
        RedAreaPos = new Vector3(RedArea.transform.position.x, 0f, RedArea.transform.position.z);
        ResetScene();
        Debug.Log("Env Start");
    }

    public void FixedUpdate() // 에피소드 종료 조건을 충족하는지 매 step 마다 확인
    {
        if (ResetComplete)
        {
            // CalculateHpReward();
            // CalculateShootReward();
            CalculateSparseReward();
            AgentGroup.AddGroupReward(StepReward);
            StepReward = 0f;
            switch (true)
            {
                case true when (ResetTimer >= MaxEnvironmentSteps && MaxEnvironmentSteps > 0):

                    Debug.Log("Episode Interrupted: Agent " + count.CurrentBlue + " vs RSA " + count.CurrentRed);
                    judgmentWin();
                    // CalculateReward();
                    var result = previousAgenthp - previousRSAhp;
                    if (result < 0)
                    {
                        count.RedWin = 1;
                        count.BlueWin = -1;
                        AgentGroup.AddGroupReward(-0.7f * WinReward * neg_scale);
                        // AgentGroup.AddGroupReward(-EpisodeReward);
                    }
                    else if (result > 0)
                    {
                        count.BlueWin = 1;
                        count.RedWin = -1;
                        AgentGroup.AddGroupReward(0.7f * WinReward);
                        // SAgentGroup.AddGroupReward(EpisodeReward);
                    }
                    
                    AgentGroup.EndGroupEpisode();
                    ResetScene();
                    break;
                case true when (count.CurrentRed == 0 && count.CurrentBlue == 0):
                    Debug.Log("Draw");
                    AgentGroup.EndGroupEpisode();
                    ResetScene();
                    break;
                case true when (count.CurrentRed == 0):
                    Debug.Log("Agent Blue Win: Agent" + count.CurrentBlue + " vs RSA " + count.CurrentRed);
                    // AgentGroup.AddGroupReward(DefaultReward);
                    count.BlueWin = 1;
                    count.RedWin = -1;
                    AgentGroup.AddGroupReward(WinReward);
                    AgentGroup.EndGroupEpisode();
                    ResetScene();
                    // GC.Collect();
                    // GC.WaitForPendingFinalizers();
                    break;
                case true when (count.CurrentBlue == 0):
                    Debug.Log("RSA Red win: Agent" + count.CurrentBlue + " vs RSA " + count.CurrentRed);
                    // AgentGroup.AddGroupReward(-DefaultReward);
                    count.RedWin = 1;
                    count.BlueWin = -1;
                    AgentGroup.AddGroupReward(-WinReward * neg_scale);
                    AgentGroup.EndGroupEpisode();
                    ResetScene();
                    // GC.Collect();
                    // GC.WaitForPendingFinalizers();
                    break;
                default:
                    ResetTimer++;
                    break;
            }
        }
    }

    public void ResetScene() // 에피소드가 리셋 될때마다 호출. 위치 등 초기화 해주는 부분
    {
        ResetComplete = false;
        StepReward = 0f;
        ResetTimer = 0;
        count.BlueWin = 0;
        count.RedWin = 0;
        count.CurrentBlue = count.BlueCount;
        count.CurrentRed = count.RedCount;
        foreach (var agent in count.AgentList)
        {
            // ResetAgent 호출 시 BlueArea의 Transform과 BlueAreaPos 전달
            ResetAgent(agent, "MARLagent", BlueArea.transform, BlueAreaPos);
        }
        foreach (var enemy in count.EnemyList)
        {
            // ResetRSA 호출 시 RedArea의 Transform과 RedAreaPos 전달
            ResetRSA(enemy, "RSA", RedArea.transform, RedAreaPos);
        }
        previousAgenthp = agent_max;
        previousRSAhp = rsa_max;
        ResetComplete = true;
    }

    // ResetAgent 함수 시그니처 및 내부 로직 변경
    public void ResetAgent(rsa_MARLagent agent, string tag, Transform area, Vector3 area_position)
    {
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
        agent.IsActive = true;
        agent.Stepdied = false;

        Rigidbody rigidbody = agent.gameObject.GetComponent<Rigidbody>();
        if(rigidbody != null) // Rigidbody가 있는 경우에만 설정
        {
            rigidbody.mass = 5.0f;
            rigidbody.velocity = Vector3.zero; // 속도 초기화 추가
            rigidbody.angularVelocity = Vector3.zero; // 회전 속도 초기화 추가
        }
    }

    // ResetRSA 함수 시그니처 및 내부 로직 변경
    public void ResetRSA(RSAcontrol rsa, string tag, Transform area, Vector3 area_position)
    {
        float scaleX = area.localScale.x;
        float scaleZ = area.localScale.z;
        float RandomValueX = UnityEngine.Random.Range(-scaleX / 2f, scaleX / 2f);
        float RandomValueZ = UnityEngine.Random.Range(-scaleZ / 2f, scaleZ / 2f);

        rsa.transform.SetPositionAndRotation(area_position + new Vector3(RandomValueX, rsa.InitialY, RandomValueZ), Quaternion.Euler(0, area.eulerAngles.y, 0));
        Physics.IgnoreCollision(rsa.GetComponent<Collider>(), floor.GetComponent<Collider>(), false);
        rsa.gameObject.tag = tag;
        rsa.hp = rsa.hpMax;
        rsa.remainShootingCool = 0;
        rsa.ForwardStick.GetComponent<forwardStickVIsual>().AgentDefault();
        rsa.IsActive = true;
        rsa.Stepdied = false;

        Renderer renderer = rsa.gameObject.GetComponent<Renderer>();
        if (renderer != null) { renderer.material = rsa.DefaultMaterial; }
        Rigidbody rigidbody = rsa.gameObject.GetComponent<Rigidbody>(); // RSA Rigidbody 가져오기
        if(rigidbody != null) // Rigidbody가 있는 경우에만 설정
        {
            rigidbody.mass = 5.0f;
            rigidbody.velocity = Vector3.zero; // 속도 초기화 추가
            rigidbody.angularVelocity = Vector3.zero; // 회전 속도 초기화 추가
        }
    }

    public void CalculateSparseReward()
    {
        StepReward = 0f;
        foreach (var agent in count.AgentList)
        {
            if (agent.Stepdied)
            {
                StepReward -= killRewardWeight * neg_scale * agent.rewardWeight;
                agent.Stepdied = false;
            }
        }
        foreach (var rsa in count.EnemyList)
        {
            if (rsa.Stepdied)
            {
                StepReward += killRewardWeight * rsa.rewardWeight;
                rsa.Stepdied = false;
            }
        }
    }
    public void CalculateHpReward()
    {
        float agent_hp = 0f;
        float rsa_hp = 0f;
        foreach (var agent in count.AgentList)
        {
            if (agent.hp > 0f) { agent_hp += agent.hp; }
        }
        foreach (var rsa in count.EnemyList)
        {
            if (rsa.hp > 0f) { rsa_hp += rsa.hp; }
        } 
        float rsa_damaged_ratio = (previousRSAhp - rsa_hp) / rsa_max; // rsa가 해당 step에서 잃은 체력 비율
        float agent_damaged_ratio = (previousAgenthp - agent_hp) / agent_max; // agent가 해당 step에서 잃은 체력 비율
        StepReward =  10*(rsa_damaged_ratio - (agent_damaged_ratio * 0.5f)); // 적을 죽이는 방향으로 유도하기 위해
        previousAgenthp = agent_hp;
        previousRSAhp = rsa_hp;
    }

    public void CalculateShootReward() 
    {
        float agent_hp = 0f;
        StepReward = 0f;
        float death_value = 0f;
        foreach (var agent in count.AgentList)
        {
            if (agent.StepShootingReward != 0f)
            {
                StepReward += agent.StepShootingReward * ShootRewardWeight;
                agent.StepShootingReward = 0f;
            }
            if (agent.hp > 0f) { agent_hp += agent.hp; }
            if (agent.Stepdied) 
            { 
                death_value += agent.rewardWeight * DeathPenalty; 
                agent.Stepdied = false;
            }
            // 사격한 적에 따라 가중치를 부여
        }
        float hp_value = hpPenalty * (previousAgenthp - agent_hp) / agent_max;
        StepReward -= neg_scale * (death_value + hp_value);
        previousAgenthp = agent_hp;
    }

    public void judgmentWin()
    {
        float agent_hp = 0f;
        float rsa_hp = 0f;
        foreach (var agent in count.AgentList)
        {
            if (agent.hp > 0f) { agent_hp += agent.hp; }
        }
        foreach (var rsa in count.EnemyList)
        {
            if (rsa.hp > 0f) { rsa_hp += rsa.hp; }
        }
        previousAgenthp = agent_hp;
        previousRSAhp = rsa_hp;
    }

}
