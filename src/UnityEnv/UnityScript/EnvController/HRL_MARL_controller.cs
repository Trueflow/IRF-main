using System.Collections;
using System.Collections.Generic;
using Unity.MLAgents;
using UnityEngine;

public class HRL_MARL_controller : MonoBehaviour
{
    [Tooltip("Max Environment Steps")] public int MaxEnvironmentSteps;
    private int ResetTimer = 0;
    public GameObject floor;
    public UpperAgent upper;

    [System.Serializable]
    public class LowerAgentInfo
    {
        public LowerAgent agent;
        [HideInInspector]
        public Vector3 InitialPos;
        [HideInInspector]
        public bool IsActive = true;
    }
    [System.Serializable]
    public class marlAgentInfo
    {
        public MARLagent agent;
        [HideInInspector]
        public Vector3 InitialPos;
        [HideInInspector]
        public bool IsActive = true;
    }
    public List<LowerAgentInfo> LowerAgentList = new();
    public List<marlAgentInfo> marlAgentList = new();

    [HideInInspector]
    public SimpleMultiAgentGroup hrlAgentGroup = new SimpleMultiAgentGroup();
    [HideInInspector]
    public SimpleMultiAgentGroup marlAgentGroup = new SimpleMultiAgentGroup();

    [HideInInspector] public int AgentCount;
    [HideInInspector] public int LowerCount;
    [HideInInspector] public int CurrentAgent;
    [HideInInspector] public int CurrentLower;
    // Start is called before the first frame update
    void Start()
    {
        int AgentIndex = 0;
        foreach (var lower in marlAgentList)
        {
            hrlAgentGroup.RegisterAgent(lower.agent);
            lower.InitialPos = lower.agent.gameObject.transform.position;
            lower.agent.MyIndex = AgentIndex;
            AgentIndex++;
        }
        int LowerIndex = 0;
        foreach (var agent in marlAgentList)
        {
            marlAgentGroup.RegisterAgent(agent.agent);
            agent.InitialPos = agent.agent.gameObject.transform.position;
            agent.agent.MyIndex = LowerIndex;
            LowerIndex++;
        }
        LowerCount = LowerAgentList.Count;
        AgentCount = marlAgentList.Count;
        ResetScene();
    }

    public void FixedUpdate() // ���Ǽҵ� ���� ������ �����ϴ��� �� step ���� Ȯ��
    {
        switch (true)
        {
            case true when (ResetTimer >= MaxEnvironmentSteps && MaxEnvironmentSteps > 0):
                // Debug.Log("Episode Interrupted");
                hrlAgentGroup.GroupEpisodeInterrupted();
                marlAgentGroup.GroupEpisodeInterrupted();
                ResetScene();
                break;
            case true when (CurrentAgent == 0):
                // Debug.Log("Agent Win");
                hrlAgentGroup.AddGroupReward(1f);
                marlAgentGroup.AddGroupReward(-1f);
                hrlAgentGroup.EndGroupEpisode();
                marlAgentGroup.EndGroupEpisode();
                ResetScene();
                break;
            case true when (CurrentAgent == 0):
                // Debug.Log("RSA win");
                hrlAgentGroup.AddGroupReward(-1f);
                marlAgentGroup.AddGroupReward(1f);
                hrlAgentGroup.EndGroupEpisode();
                marlAgentGroup.EndGroupEpisode();
                ResetScene();
                break;
            default:
                ResetTimer++;
                break;
        }
    }

    public void ResetScene() // ���Ǽҵ尡 ���� �ɶ����� ȣ��. ��ġ �� �ʱ�ȭ ���ִ� �κ�
    {
        CurrentAgent = AgentCount;
        CurrentLower = LowerCount;
        foreach (var lower in LowerAgentList)
        {
            ResetLower(lower);
        }
        foreach (var agent in marlAgentList)
        {
            ResetAgent(agent);
        }
        ResetTimer = 0;
    }

    public void LowerShoot(LowerAgent lower, GameObject Obj)
    {
        if (Obj.CompareTag("MARLagent"))
        {
            lower.AddReward(0.02f);
            MARLagent agent = Obj.GetComponent<MARLagent>();
            agent.hp -= lower.attack;
            if (agent.hp <= 0)
            {
                agent.ActiveFalse();
                CurrentAgent--;
            }
        }
    }
    public void AgentShoot(MARLagent agent, GameObject Obj)
    {
        if (Obj.CompareTag("LowerAgent"))
        {
            agent.AddReward(0.02f);
            LowerAgent lower = Obj.GetComponent<LowerAgent>();
            lower.hp -= agent.attack;
            if (lower.hp <= 0)
            {
                lower.ActiveFalse();
                CurrentLower--;
            }
        }
    }

    public void ResetAgent(marlAgentInfo agent)
    {
        float RandomValue = Random.Range(-0.5f, 0.5f);
        agent.agent.transform.SetPositionAndRotation(agent.InitialPos + new Vector3(-RandomValue, 0, RandomValue), Quaternion.Euler(0, RandomValue * 30f, 0));
        Physics.IgnoreCollision(agent.agent.gameObject.GetComponent<Collider>(), floor.GetComponent<Collider>(), false);
        Rigidbody rigidbody = agent.agent.gameObject.GetComponent<Rigidbody>();
        Renderer renderer = agent.agent.gameObject.GetComponent<Renderer>();
        rigidbody.mass = 1.0f;
        if (renderer != null) { renderer.material = agent.agent.DefaultMaterial; }
        agent.IsActive = true;
        agent.agent.gameObject.tag = "MARLagent";
        agent.agent.hp = agent.agent.hpMax;
    }

    public void ResetLower(LowerAgentInfo agent)
    {
        float RandomValue = Random.Range(-0.5f, 0.5f);
        agent.agent.transform.SetPositionAndRotation(agent.InitialPos + new Vector3(-RandomValue, 0, RandomValue), Quaternion.Euler(0, RandomValue * 30f, 0));
        Physics.IgnoreCollision(agent.agent.gameObject.GetComponent<Collider>(), floor.GetComponent<Collider>(), false);
        Rigidbody rigidbody = agent.agent.gameObject.GetComponent<Rigidbody>();
        Renderer renderer = agent.agent.gameObject.GetComponent<Renderer>();
        rigidbody.mass = 1.0f;
        if (renderer != null) { renderer.material = agent.agent.DefaultMaterial; }
        agent.IsActive = true;
        agent.agent.gameObject.tag = "LowerAgent";
        agent.agent.hp = agent.agent.hpMax;
    }

}
