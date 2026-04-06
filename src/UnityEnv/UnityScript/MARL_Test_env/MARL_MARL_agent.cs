using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Policies;
using Unity.MLAgents.Sensors;
using Unity.VisualScripting;
using NUnit.Framework.Constraints;
using System.Linq;
using System;
using System.Security.Cryptography;

public class MARL_MARLagent : Agent
{
    // MARLvsMARL Agent Script
    public enum Position
    {
        Infantry = 0, // Infantry
        Armored = 1, // Armored
        Artillery = 2 // Artillery
    }
    public ObjectCount count;
    public int MyTeam;
    // public RSA_marl_Controller EnvController;

    // public List<MARLagent> EnemyList;
    // public List<RSAcontrol> EnemyList;
    [HideInInspector] public int MyIndex;

    [HideInInspector] public BehaviorParameters Behavior;
    [HideInInspector] public Rigidbody RigidBody;
    public Position position;
    [HideInInspector] public int myposition;

    // Agent Die or Alive Textures
    public Material DefaultMaterial;
    public Material DisabledMaterial;
    public GameObject ForwardStick;
    public LineRenderer AttackLineRender;
    [HideInInspector] public Vector3 AttackLineBase = new Vector3(0f, -1f, 0f);

    // Agent Move Speed, Rotation Range, Shooting Range, Reward Weight
    [HideInInspector] float MoveSpeed;
    [HideInInspector] float RotationRange;
    [HideInInspector] float ShootingRange;
    [HideInInspector] public float rewardWeight = 1f;

    [HideInInspector] public float attack;
    [HideInInspector] public float hpMax;
    [HideInInspector] public float hp;
    [HideInInspector] public float InitialY;

    [HideInInspector] public int remainShootingCool = 0;
    [HideInInspector] public int ShootingCooldownTerm = 30;
    public float AttackAngle = 20f;

    [HideInInspector] public bool IsActive = true;
    [HideInInspector] public List<float> ActionMask;
    [HideInInspector] public bool CanAttack;
    [HideInInspector] public GameObject target;
    public List<MARL_MARLagent> EnemyList;

    public LayerMask EnemyLayer;
    public LayerMask EnvObjectLayer;

    public float StepShootingReward;
    public List<float> TargetList;
    [HideInInspector] public bool Stepdied = false;

    [HideInInspector] public float env_threshold_x;
    [HideInInspector] public float env_threshold_z;

    public override void Initialize()
    {
        if (AttackLineRender != null) { AttackLineRender.enabled = false; }
        RigidBody = GetComponent<Rigidbody>();
        switch (position) // Infantry, Armored, Arillery에 따른 속성 설정
        {
            case (Position)0: // Infantry
                ShootingRange = 5f;
                attack = 10f;
                MoveSpeed = 1f;
                RotationRange = 5f;
                rewardWeight = 0.3f;
                hpMax = 200f;
                InitialY = -1f;
                ShootingCooldownTerm = 20;
                break;
            case (Position)1: // Armored
                ShootingRange = 10f;
                attack = 100f;
                MoveSpeed = 2f;
                RotationRange = 1f;
                rewardWeight = 1f;
                hpMax = 1000f;
                InitialY = -1f;
                ShootingCooldownTerm = 50;
                break;
            case (Position)2: // Artillery
                ShootingRange = 30f;
                attack = 100f;
                MoveSpeed = 0.4f;
                RotationRange = 1f;
                rewardWeight = 0.8f;
                hpMax = 600f;
                InitialY = -1f;
                ShootingCooldownTerm = 250;
                break;
        }
        myposition = (int)position;
        env_threshold_x = count.floor.transform.localScale.x / 2f;
        env_threshold_z = count.floor.transform.localScale.z / 2f;
    }
    public void GetActionMask() // agent action mask (1f: available, 0f: unavailable)
    {
        if (!IsActive) 
        { 
            // only default action is available
            ActionMask = new() { 1f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f };
            if (!count.AttackLeastDistance)
            {
                for (int i = 1; i < count.RedCount; i++) { ActionMask.Add(0f); }
            }
        }
        else
        {
            ActionMask = new() { 0f, 1f, 1f, 1f, 1f, 1f, 1f, 1f }; // without attack 
            getMoveMask(); // action mask - move (2~8)
            getAttackMask(); // action mask - attack (leastdist - 1 or 0, choose - num_agents {1 or 0})
        }
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        // My observation space size : 6
        sensor.AddObservation(transform.forward); // forward
        sensor.AddObservation(ShootingRange); // shooting range
        sensor.AddObservation(attack); // attack
        sensor.AddObservation(hp); // hp
        foreach (var enemy in EnemyList)
        {
            if (enemy.IsActive) // if the enemy is active
            { // enemy observation space size : 10 * num_agents
                var toEnemy = enemy.gameObject.transform.position - transform.position;
                sensor.AddObservation(toEnemy.normalized); // normalized direction
                sensor.AddObservation(toEnemy.magnitude); // distance
                sensor.AddObservation(enemy.transform.forward); // forward
                sensor.AddObservation(enemy.ShootingRange); // shooting range
                sensor.AddObservation((float)(int)enemy.attack);
                sensor.AddObservation((float)(int)enemy.hp); // hp
            }
            else // if the enemy is not active (padding)
            {
                sensor.AddObservation(Vector3.zero);
                sensor.AddObservation(0);
                sensor.AddObservation(Vector3.zero);
                sensor.AddObservation(0);
                sensor.AddObservation(0);
                sensor.AddObservation(0);
            }
        }
        // python mlagents actionmask
        sensor.AddObservation(ActionMask); // 9 or 8+num_agents : actionmask size
        if (MyTeam == 0) { sensor.AddObservation(count.BlueWin); }
        else { sensor.AddObservation(count.RedWin); }
        sensor.AddObservation(IsActive ? 1 : 0); // active or not
    } // 6 + 10 * num_agents + 8+num_agents + 2 = 16 + 11 * num_agents (if choose enemy)

    public override void OnActionReceived(ActionBuffers actionBuffers)
    {
        var action = actionBuffers.DiscreteActions[0];
        // RigidBody.velocity = Vector3.zero;
        // RigidBody.angularVelocity = Vector3.zero;
        var forward = Vector3.zero;
        var rotate = Vector3.zero;
        switch (action)
        {
            case 0: break; // no action - if agent died
            case 1: break; // no action - alive agent
            case 2:
                forward = transform.forward;;
                break;
            case 3:
                forward = -transform.forward;
                break;
            case 4:
                forward = transform.right;
                break;
            case 5:
                forward = -transform.right;
                break;
            case 6:
                rotate = transform.up;;
                break;
            case 7:
                rotate = -transform.up;
                break;
            default:
                if (!count.AttackLeastDistance)
                {
                    var targetIndex = action - 8;
                    if (targetIndex < count.RedCount) { target = EnemyList[targetIndex].gameObject; }
                }
                if (target != null) { AgentShoot(this, target); }
                break;
        }
        RigidBody.AddForce(MoveSpeed * forward, ForceMode.VelocityChange);
        transform.Rotate(rotate, RotationRange);
    }

    public void AgentShoot(MARL_MARLagent agent, GameObject Obj)
    {
        MARL_MARLagent enemy = Obj.GetComponent<MARL_MARLagent>();
        enemy.hp -= agent.attack;
        if (AttackLineRender != null) { StartCoroutine(AttackLine(transform.position, Obj.transform.position, 0.2f)); }
        if (enemy.hp <= 0)
        {
            enemy.ActiveFalse();
        }
        remainShootingCool = ShootingCooldownTerm;
        ForwardStick.GetComponent<forwardStickVIsual>().AgentShoot();
        if (count.ShootReward) { StepShootingReward = count.ShootParam * enemy.rewardWeight * attack/(float)ShootingCooldownTerm; }
    }

    // Update is called once per frame
    public void FixedUpdate()
    {
        GetActionMask();
        // RequestDecision: CollectObservation -> OnActionReceived
        RequestDecision();

        if (IsActive)
        {
            if (Mathf.Abs(transform.position.z) > env_threshold_z || Mathf.Abs(transform.position.x) > env_threshold_x || transform.position.y < -5f)
            {
                ActiveFalse();
                transform.SetPositionAndRotation(Vector3.zero, Quaternion.Euler(0f, 0f, 0f));
            }

            Vector3 CurrentAngle = transform.rotation.eulerAngles;
            if (Mathf.Abs(CurrentAngle.x) > 5f || Mathf.Abs(CurrentAngle.z) > 5f)
            {
                transform.rotation = Quaternion.Euler(0f, CurrentAngle.y, 0f);
            }
        }
    }

    public void getMoveMask()
    {
        float leastDistance = 2f;
        bool frontBlocked = Physics.Raycast(transform.position, transform.forward, leastDistance, EnvObjectLayer);
        bool backBlocked = Physics.Raycast(transform.position, -transform.forward, leastDistance, EnvObjectLayer);
        bool rightBlocked = Physics.Raycast(transform.position, transform.right, leastDistance, EnvObjectLayer);
        bool leftBlocked = Physics.Raycast(transform.position, -transform.right, leastDistance, EnvObjectLayer);

        if (frontBlocked) { ActionMask[2] = 0f; }
        if (backBlocked) { ActionMask[3] = 0f; }
        if (rightBlocked) { ActionMask[4] = 0f; }
        if (leftBlocked) { ActionMask[5] = 0f; }
    }

    public void getAttackMask()
    {
        if (remainShootingCool > 0) // shooting cooldown check
        {
            if (count.AttackLeastDistance)
            {
                ActionMask.Add(0f);
            }
            else
            {
                for (int i = 0; i < count.RedCount; i++)
                {
                    ActionMask.Add(0f);
                }
            }
            remainShootingCool--;
            if (remainShootingCool == 0) { ForwardStick.GetComponent<forwardStickVIsual>().AgentDefault(); }
        }
        else
        {
            if (count.AttackLeastDistance)
            {
                AttackCheck();
                if (CanAttack)
                {
                    ActionMask.Add(1f);
                    // ForwardStick.GetComponent<forwardStickVIsual>().AgentDefault();
                }
                else { ActionMask.Add(0f); }
            }
            else
            {
                AttackCheckList();
                foreach (var AttackPossibility in TargetList)
                {
                    ActionMask.Add(AttackPossibility);
                }
            }
        }
        
    }

    public void AttackCheck()
    {
        CanAttack = false;
        target = null;
        float nearest = -1f;
        Collider[] hitColliders = Physics.OverlapSphere(transform.position, ShootingRange, EnemyLayer);
        // all opponents are in EnemyLayer
        foreach (Collider col in hitColliders)
        {
            Vector3 TargetDirection = (col.transform.position - transform.position).normalized;
            float angle = Vector3.Angle(transform.forward, TargetDirection);
            if (angle <= AttackAngle && col.tag != "Obstacle")
            {
                // check if the target is in the attack angle and not an obstacle
                Vector3 start_position = transform.position + Vector3.up; // start position
                if (Physics.Raycast(start_position, TargetDirection, out RaycastHit hit, ShootingRange, EnvObjectLayer))
                {
                    float distance = Vector3.Distance(hit.transform.position, transform.position);
                    if (nearest == -1f || distance < nearest)
                    {
                        nearest = distance;
                        target = hit.collider.gameObject;
                        CanAttack = true;
                    }
                }
            }
        }
    }

    public void AttackCheckList()
    {
        CanAttack = false;
        TargetList = Enumerable.Repeat(0f, count.RedCount).ToList();

        Collider[] hitColliders = Physics.OverlapSphere(transform.position, ShootingRange, EnemyLayer);

        foreach (Collider col in hitColliders)
        {
            Vector3 TargetDirection = (col.transform.position - transform.position).normalized;
            float angle = Vector3.Angle(transform.forward, TargetDirection);
            if (angle <= AttackAngle && col.tag != ("Obstacle"))
            {
                // check if the target is in the attack angle and not an obstacle
                Vector3 start_position = transform.position + Vector3.up;
                if (Physics.Raycast(start_position, TargetDirection, out RaycastHit hit, ShootingRange, EnvObjectLayer))
                {
                    int enemyIndex = EnemyList.IndexOf(col.GetComponent<MARL_MARLagent>());
                    if (enemyIndex != -1) { TargetList[enemyIndex] = 1f; }
                }
            }
        }
    }

    public void ActiveFalse()
    {
        if (!IsActive) return;
        hp = 0f;
        //Debug.Log("Agent ActiveFalse called");
        IsActive = false;
        if (gameObject.CompareTag("BlueTeam")) { count.CurrentBlue--; }
        else if (gameObject.CompareTag("RedTeam")) { count.CurrentRed--; }
        gameObject.tag = "Obstacle"; // ???? ????????? ???? ?????? ??
        // SkinnedMeshRenderer renderer = gameObject.transform.GetChild(1).gameObject.GetComponent<SkinnedMeshRenderer>();
        Renderer renderer = GetComponent<Renderer>();
        if (renderer != null) { renderer.material = DisabledMaterial; }
        Collider floorCollider = count.floor.GetComponent<Collider>();
        Physics.IgnoreCollision(GetComponent<Collider>(), floorCollider, true);
        transform.position = new Vector3(transform.position.x, transform.position.y - 1.3f, transform.position.z);
        RigidBody.mass = 20; // ???? ????????? ?????????
        RigidBody.velocity = Vector3.zero;
        RigidBody.angularVelocity = Vector3.zero;
        Stepdied = true;
    }

    public IEnumerator AttackLine(Vector3 Start, Vector3 End, float duration)
    {
        // Vector3 StartPos = Start + AttackLineBase;
        Start.y = AttackLineBase.y;
        End.y = AttackLineBase.y;
        if (AttackLineRender != null)
        {
            AttackLineRender.SetPosition(0, Start);
            AttackLineRender.SetPosition(1, End);
            AttackLineRender.enabled = true;
            yield return new WaitForSeconds(duration);
            AttackLineRender.enabled = false;
        }
        yield return null;
    }

    public override void Heuristic(in ActionBuffers actionsOut)
    {
        var discreteActions = actionsOut.DiscreteActions;

        // ????(??? ????? ???? ??)?? "1?? ???(=No Action, alive)"???? ????
        // (????? ???? "0?? ???"?? ???????? ?? ???? ??????, 
        //  ???? 0???? '???/?????' ???·? ?????? ??????? ??? 1???? ???????? ????.)
        discreteActions[0] = 1;

        // ? ????? ??????, ??? ???? ?? ??? ????
        // 2: forward, 3: backward, 4: right, 5: left, 6: rotate left, 7: rotate right, 8: attack
        // ?? ????? ??? ??????? ???? ????????.

        if (Input.GetKey(KeyCode.W))
        {
            discreteActions[0] = 2; // ????
        }
        else if (Input.GetKey(KeyCode.S))
        {
            discreteActions[0] = 3; // ????
        }
        else if (Input.GetKey(KeyCode.D))
        {
            discreteActions[0] = 4; // ?????? ???
        }
        else if (Input.GetKey(KeyCode.A))
        {
            discreteActions[0] = 5; // ???? ???
        }
        else if (Input.GetKey(KeyCode.Q))
        {
            discreteActions[0] = 6; // ???? ???
        }
        else if (Input.GetKey(KeyCode.E))
        {
            discreteActions[0] = 7; // ?????? ???
        }
        else if (Input.GetKey(KeyCode.Space))
        {
            discreteActions[0] = 8; // ????
        }
    }
}
