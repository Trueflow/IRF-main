using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.MLAgents.Policies;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using JetBrains.Annotations;
using System;
using System.Linq;

public class RSA_marl_Control_2 : MonoBehaviour
{
    // MARL vs RSA Environment Controller
    [Tooltip("Max Environment Steps")] public int MaxEnvironmentSteps;
    private int ResetTimer = 0;

    public GameObject floor;
    public GameObject BlueArea;
    public GameObject RedArea;
    [HideInInspector] public Vector3 BlueAreaTransform;
    [HideInInspector] public Vector3 RedAreaTransform;
    public GameObject blueOccupiedArea;
    public GameObject redOccupiedArea;
    public ObjectCount count;

    [HideInInspector]
    public SimpleMultiAgentGroup AgentGroup = new SimpleMultiAgentGroup();

    int AgentOccupiedCount=0;
    int RSAOccupiedCount=0;


    void Start() // 최초 실행 시 호출됨. 환경에 대한 기본 설정
    {
        float BlueX = BlueArea.transform.position.x;
        float BlueZ = BlueArea.transform.position.z;
        float RedX = RedArea.transform.position.x;
        float RedZ = RedArea.transform.position.z;
        BlueAreaTransform = new Vector3 (BlueX, 0f, BlueZ);
        RedAreaTransform = new Vector3 (RedX, 0f, RedZ);
        int AgentIndex = 0;
        foreach (var agent in count.AgentList)
        {
            AgentGroup.RegisterAgent(agent);
            agent.MyIndex = AgentIndex;
            AgentIndex++;
        }
        int EnemyIndex = 0;
        foreach (var enemy in count.EnemyList)
        {
            enemy.MyIndex = EnemyIndex;
            EnemyIndex++;
        }
        ResetScene();
    }

    public void FixedUpdate() // 에피소드 종료 조건을 충족하는지 매 step 마다 확인
    {
        OccupiedCheck(blueOccupiedArea, "MARLagent", AgentOccupiedCount);
        OccupiedCheck(redOccupiedArea, "RSA", RSAOccupiedCount);

        switch(true)
        {
            case true when (ResetTimer < 50):
                ResetTimer++;
                break;
            case true when (ResetTimer >= MaxEnvironmentSteps && MaxEnvironmentSteps > 0):
                Debug.Log("Episode Interrupted: Occupied Agents - "+ AgentOccupiedCount);
                AgentGroup.AddGroupReward(1 * AgentOccupiedCount);
                AgentGroup.EndGroupEpisode();
                ResetScene();
                // GC.Collect();
                // GC.WaitForPendingFinalizers();
                break;
            case true when (AgentOccupiedCount >= 3):
                Debug.Log("Agent Occupied RSA area");
                AgentGroup.AddGroupReward(3);
                AgentGroup.EndGroupEpisode();
                ResetScene();
                // GC.Collect();
                // GC.WaitForPendingFinalizers();
                break;
            case true when (RSAOccupiedCount >= 3):
                Debug.Log("RSA Occupied Agent area");
                AgentGroup.AddGroupReward(-1);
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

    public void ResetScene() // 에피소드가 리셋 될때마다 호출. 위치 등 초기화 해주는 부분
    {
        Debug.Log("Reset Scene method called");
        AgentOccupiedCount = 0;
        RSAOccupiedCount = 0;
        foreach (var agent in count.AgentList)
        {
            ResetAgent(agent, "MARLagent", BlueAreaTransform);
        }
        foreach (var enemy in count.EnemyList)
        {
            ResetRSA(enemy, "RSA", RedAreaTransform);
        }
        ResetTimer = 0;
    }

    public void ResetAgent(rsa_MARLagent agent, string tag, Vector3 InitialPos)
    {
        float RandomValueX = UnityEngine.Random.Range(-10f, 10f);
        float RandomValueZ = UnityEngine.Random.Range(-10f, 10f);
        agent.transform.SetPositionAndRotation(InitialPos + new Vector3(RandomValueX, agent.InitialY, RandomValueZ), Quaternion.Euler(0, 45f, 0));
        Physics.IgnoreCollision(agent.gameObject.GetComponent<Collider>(), floor.GetComponent<Collider>(), false);
        Rigidbody rigidbody = agent.gameObject.GetComponent<Rigidbody>();
        // Renderer renderer = agent.gameObject.GetComponent<Renderer>();
        // if (renderer != null) { renderer.material = agent.DefaultMaterial; }
        rigidbody.mass = 1.0f;
        agent.gameObject.tag = tag;
        agent.hp = agent.hpMax;
        agent.IsActive = true;
    }

    public void ResetRSA(RSAcontrol rsa,  string tag, Vector3 InitialPos)
    {
        float RandomValueX = UnityEngine.Random.Range(-10f, 10f);
        float RandomValueZ = UnityEngine.Random.Range(-10f, 10f);
        rsa.transform.SetPositionAndRotation(InitialPos + new Vector3(RandomValueX, rsa.InitialY, RandomValueZ), Quaternion.Euler(0, 225f, 0));
        Physics.IgnoreCollision(rsa.GetComponent<Collider>(), floor.GetComponent<Collider>(), false);
        Rigidbody rigidbody = rsa.gameObject.GetComponent<Rigidbody>();
        // RSAcontrol rsa = enemy.enemy.GetComponent<RSAcontrol>();
        // Renderer renderer = rsa.gameObject.GetComponent<Renderer>();
        // if (renderer != null) { renderer.material = rsa.DefaultMaterial; }
        rigidbody.mass = 1.0f;
        rsa.gameObject.tag = tag;
        rsa.IsActive = true;
        rsa.hp = rsa.hpMax;
    }

    public void OccupiedCheck(GameObject Obj, string tag, int ObjectCount)
    {
        Collider zoneCollider = blueOccupiedArea.GetComponent<BoxCollider>();
        GameObject[] allObjects = GameObject.FindGameObjectsWithTag(tag);
        int count = 0;
        foreach (GameObject agent in allObjects)
        {
            if (zoneCollider.bounds.Intersects(agent.GetComponent<Collider>().bounds))
            {
                count++;
            }
        }
        ObjectCount = count;
    }

}
