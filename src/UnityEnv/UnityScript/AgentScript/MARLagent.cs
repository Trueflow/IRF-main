using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Policies;
using Unity.MLAgents.Sensors;


public class MARLagent : Agent
{
    // RSA 시나리오 에서만 사용합니다.
    public enum Position
    {
        Infantry = 0, //보병
        Armored = 1, // 기갑 - 벽 못뚫음
        Tank = 2 // 포병 - 벽 뚫음
    }
    public ObjectCount count;
    // public RSA_marl_Controller EnvController;

    // public List<MARLagent> EnemyList;
    // public List<RSAcontrol> EnemyList;
    [HideInInspector] public int MyIndex;

    [HideInInspector] public BehaviorParameters Behavior;
    [HideInInspector] public Rigidbody RigidBody;
    public Position position;

    // 에이전트의 활성화 여부를 시각적으로 표시하기 위한 용도로 추가했습니다.
    public Material DefaultMaterial;
    public Material DisabledMaterial;

    // 에이전트의 속성에 관련된 변수입니다.
    [HideInInspector] float MoveSpeed;
    [HideInInspector] float RotationRange;
    [HideInInspector] float ShootingRange;
    [HideInInspector] public float rewardWeight = 1f;

    [HideInInspector] public float attack;
    [HideInInspector] public float hpMax;
    [HideInInspector] public float hp;
    [HideInInspector] public float InitialY;

    [HideInInspector] public int ShootingTerm = 0;

    [HideInInspector] public bool IsActive = true;


    public LayerMask RaycastLayer;
    [Tooltip("Selecting will turn on action masking. Note that a model trained with action " +
        "masking turned on may not behave optimally when action masking is turned off.")]
    public bool maskActions = true;

    public override void Initialize()
    {
        RigidBody = GetComponent<Rigidbody>();
        switch (position)
        {
            case (Position)0: // 보병
                ShootingRange = 5f;
                attack = 30f;
                MoveSpeed = 1f;
                RotationRange = 5f;
                rewardWeight = 1f;
                hpMax = 100f;
                InitialY = -0.9f;
                break;
            case (Position)1: // 기갑
                ShootingRange = 10f;
                attack = 100f;
                MoveSpeed = 2f;
                RotationRange = 3f;
                rewardWeight = 10f;
                hpMax = 1000f;
                InitialY = -1f;
                break;
            case (Position)2: // 탱크
                MoveSpeed = 0.2f;
                RotationRange = 10f;
                ShootingRange = 10f;
                attack = 60f;
                hpMax = 300f;
                break;
        }
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        foreach (var enemy in count.EnemyList)
        {
            if (enemy.IsActive) // 적이 생존해있는 경우
            {
                var toEnemy = enemy.gameObject.transform.position - transform.position;
                sensor.AddObservation(toEnemy.normalized); // 적의 방향
                sensor.AddObservation(toEnemy.magnitude); // 적과의 거리
            }
            else // 적이 사망한 경우
            {
                sensor.AddObservation(Vector3.zero);
                sensor.AddObservation(0);
            }
        }
        sensor.AddObservation(hp);
        sensor.AddObservation(count.BlueWin);
        sensor.AddObservation(IsActive ? 1 : 0);
    }
    public override void WriteDiscreteActionMask(IDiscreteActionMask actionMask)
    {
        // OnActionRecieved 전에 실행되는 메소드입니다.
        // 에이전트가 할 수 없는 행동을 비활성화 시키는 단계입니다.
        if (!IsActive)
        {
            // branch 0 : move
            actionMask.SetActionEnabled(0, 1, false);
            actionMask.SetActionEnabled(0, 2, false);
            actionMask.SetActionEnabled(0, 3, false);
            actionMask.SetActionEnabled(0, 4, false);
        }
        else
        {
            actionMask.SetActionEnabled(0, 0, false);
        }
    }

    public override void OnActionReceived(ActionBuffers actionBuffers)
    {
        var move = actionBuffers.DiscreteActions[0];
        // Vector3 forward = new Vector3 (0.707f, 0f, 0.707f);

        switch (move)
        {
            case 0:
                break; // no action
            case 1:
                RigidBody.AddForce(MoveSpeed * transform.forward, ForceMode.VelocityChange);
                break;
            case 2:
                RigidBody.AddForce(-MoveSpeed * transform.forward, ForceMode.VelocityChange);
                break;
            case 3:
                RigidBody.AddTorque(Vector3.up * RotationRange, ForceMode.VelocityChange);
                // transform.Rotate(transform.up, RotationRange * Time.deltaTime, 0);
                break;
            case 4:
                RigidBody.AddTorque(Vector3.down * RotationRange, ForceMode.VelocityChange);
                // transform.Rotate(transform.up, -RotationRange * Time.deltaTime, 0);
                break;
        }
    }

    public void OnCollisionEnter(Collision collision)
    {
        if (collision.collider.CompareTag("Wall"))
        {
            // Debug.Log("Collision Detected, rotate the agent");
            transform.Rotate(0, 180f, 0);
            RigidBody.velocity = Vector3.zero;
        }
    }

    public void ActiveFalse()
    {
        //Debug.Log("Agent ActiveFalse called");
        IsActive = false;
        gameObject.tag = "DiedObj";
        // SkinnedMeshRenderer renderer = gameObject.transform.GetChild(1).gameObject.GetComponent<SkinnedMeshRenderer>();
        Renderer renderer = GetComponent<Renderer>();
        if (renderer != null)
        {
            renderer.material = DisabledMaterial;
        }
        Collider floorCollider = count.floor.GetComponent<Collider>();
        Physics.IgnoreCollision(GetComponent<Collider>(), floorCollider, true);
        transform.position = new Vector3(transform.position.x, transform.position.y - 0.6f, transform.position.z);
        RigidBody.mass = 100; // 죽은 에이전트가 못움직이게
        RigidBody.velocity = Vector3.zero;
        RigidBody.angularVelocity = Vector3.zero;
    }

    public void AgentShoot(MARLagent agent, GameObject Obj)
    {
        if (Obj.CompareTag("RSA"))
        {
            agent.AddReward(0.02f);
            RSAcontrol rsa = Obj.GetComponent<RSAcontrol>();
            rsa.hp -= agent.attack;
            if (rsa.hp <= 0)
            {
                rsa.ActiveFalse();
                agent.AddReward(rsa.rewardWeight);
                count.CurrentRed--;
            }
        }
    }

    // Update is called once per frame
    public void FixedUpdate()
    {
        if (maskActions)
        { RequestDecision(); }

        if (ShootingTerm > 0)
        {
            ShootingTerm--;
        }
        if (IsActive)
        {
            //Debug.DrawRay(transform.position, transform.forward * ShootingRange, Color.blue);
            if (Physics.Raycast(transform.position, transform.forward, out RaycastHit hit, ShootingRange, RaycastLayer) && ShootingTerm == 0)
            {
                //Debug.DrawRay(transform.position, transform.forward * ShootingRange, Color.red);
                AgentShoot(this, hit.collider.gameObject);
                ShootingTerm = 5;
            }
            if (Mathf.Abs(transform.position.z) > 100f | Mathf.Abs(transform.position.x) > 100f)
            {
                ActiveFalse();
                count.CurrentBlue--;
                transform.SetPositionAndRotation(Vector3.zero, Quaternion.Euler(0f, 0f, 0f));
            }
        }
    }
}
