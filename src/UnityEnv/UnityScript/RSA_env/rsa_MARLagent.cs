using System.Collections;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Policies;
using Unity.MLAgents.Sensors;
using Unity.VisualScripting;
using NUnit.Framework.Constraints;
using System;
using System.Security.Cryptography;

public class rsa_MARLagent : Agent
{
    // RSA Scenario Agent Script (updated comment)
    public enum Position
    {
        Infantry = 0, // Infantry (updated comment)
        Armored = 1, // Armored (updated comment)
        Artillery = 2 // Artillery (updated comment, though original was Tank, MARL_MARL uses Artillery for 2)
    }
    public ObjectCount count;
    // MyTeam variable is not in rsa_MARLagent, keeping it that way.
    // public RSA_marl_Controller EnvController;

    [HideInInspector] public int MyIndex;

    [HideInInspector] public BehaviorParameters Behavior;
    [HideInInspector] public Rigidbody RigidBody;
    public Position position;
    [HideInInspector] public int myposition;

    // Agent Die or Alive Textures (updated comment)
    public Material DefaultMaterial;
    public Material DisabledMaterial;
    public GameObject ForwardStick;
    public LineRenderer AttackLineRender; // Added from MARL_MARL_agent.cs
    [HideInInspector] public Vector3 AttackLineBase = new Vector3(0f, -1f, 0f); // Added from MARL_MARL_agent.cs

    // Agent Move Speed, Rotation Range, Shooting Range, Reward Weight (updated comment)
    [HideInInspector] float MoveSpeed;
    [HideInInspector] float RotationRange;
    [HideInInspector] float ShootingRange;
    [HideInInspector] public float rewardWeight = 1f;

    [HideInInspector] public float attack;
    [HideInInspector] public float hpMax;
    [HideInInspector] public float hp;
    [HideInInspector] public float InitialY; // Standardized initial Y

    [HideInInspector] public int remainShootingCool = 0;
    [HideInInspector] public int ShootingCooldownTerm = 30; // Standardized from MARL_MARL_agent.cs
    public float AttackAngle = 20f; // Standardized from MARL_MARL_agent.cs

    [HideInInspector] public bool IsActive = true;
    [HideInInspector] public List<float> ActionMask;
    [HideInInspector] public bool CanAttack;
    public GameObject target; // target is GameObject in rsa, List<MARL_MARLagent> in MARL_MARL. Keeping GameObject for RSA.

    // Removed SkinnedMeshRenderer render;
    // Removed List<float> MovingRange;
    public LayerMask EnemyLayer;
    public LayerMask EnvObjectLayer;
    // Removed Animator animator;


    public float StepShootingReward;
    public List<float> TargetList; // This is List<float> in rsa, List<MARL_MARLagent> in MARL_MARL. Keeping List<float>.
    [HideInInspector] public bool Stepdied = false; // Standardized from MARL_MARL_agent.cs

    [HideInInspector] public float env_threshold_x; // Added from MARL_MARL_agent.cs
    [HideInInspector] public float env_threshold_z; // Added from MARL_MARL_agent.cs

    public override void Initialize()
    {
        if (AttackLineRender != null) { AttackLineRender.enabled = false; }
        RigidBody = GetComponent<Rigidbody>();
        switch (position)
        {
            case Position.Infantry: // Infantry
                ShootingRange = 5f;
                attack = 10f;
                MoveSpeed = 4f; // 2f → 4f로 증가 (RSA와 동일)
                RotationRange = 5f;
                rewardWeight = 0.4f;
                hpMax = 150f;
                ShootingCooldownTerm = 5; 
                InitialY = -1f;
                break;
            case Position.Armored: // Armored
                ShootingRange = 10f;
                attack = 50f;
                MoveSpeed = 8f; // 4f → 8f로 증가 (RSA와 동일)
                RotationRange = 2.5f;
                rewardWeight = 1f;
                hpMax = 500f;
                ShootingCooldownTerm = 20;
                InitialY = -1f;
                break;
            case Position.Artillery: // Artillery (Tank)
                ShootingRange = 20f;
                attack = 50f;
                MoveSpeed = 6f; // 3f → 6f로 증가 (RSA와 동일)
                RotationRange = 1f;
                rewardWeight = 0.8f;
                hpMax = 600f;
                ShootingCooldownTerm = 50;
                InitialY = -1f;
                break;
        }
        /*switch (position)
        {
            case Position.Infantry: // Infantry
                ShootingRange = 5f;
                attack = 10f;
                MoveSpeed = 1f;
                RotationRange = 5f;
                rewardWeight = 0.3f;
                hpMax = 200f;
                ShootingCooldownTerm = 20; 
                InitialY = -1f;
                break;
            case Position.Armored: // Armored
                ShootingRange = 10f;
                attack = 100f;
                MoveSpeed = 2f;
                RotationRange = 2f;
                rewardWeight = 1f;
                hpMax = 1000f;
                ShootingCooldownTerm = 50;
                InitialY = -1f;
                break;
            case Position.Artillery: // Artillery (Tank)
                ShootingRange = 30f;
                attack = 100f;
                MoveSpeed = 0.4f;
                RotationRange = 1f;
                rewardWeight = 0.2f;
                hpMax = 600f;
                ShootingCooldownTerm = 250;
                InitialY = -1f;
                break;
        }*/
        
        myposition = (int)position;
        env_threshold_x = count.floor.transform.localScale.x / 2f; // Added from MARL_MARL_agent.cs
        env_threshold_z = count.floor.transform.localScale.z / 2f; // Added from MARL_MARL_agent.cs
    }

     public void GetActionMask() // agent action mask (1f: available, 0f: unavailable) (updated comment)
    {
        if (!IsActive) 
        { 
            // only default action is available (updated comment)
            ActionMask = new() { 1f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f };
            // RSA agent might have different enemy count logic, ensure count.RedCount is appropriate here
            if (!count.AttackLeastDistance)
            {
                // Assuming RedCount refers to the number of RSA enemies
                for (int i = 1; i < count.RedCount; i++) { ActionMask.Add(0f); } 
            }
        }
        else
        {
            ActionMask = new() { 0f, 1f, 1f, 1f, 1f, 1f, 1f, 1f }; // without attack (updated comment)
            getMoveMask(); // Changed from CollisionCheck()
            getAttackMask(); // New function to encapsulate attack masking logic
        }
        // Debug.Log("ActionMask: " + string.Join(", ", ActionMask));
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        // Debug.Log("CollectObservations called");
        // sensor.AddObservation(transform.position); // 내 위치
        sensor.AddObservation(transform.forward);
        sensor.AddObservation(ShootingRange);
        sensor.AddObservation(attack);
        sensor.AddObservation(hp); // 남은 체력 // 6개
        foreach (var enemy in count.EnemyList)
        {
            if (enemy.IsActive) // 적이 생존해있는 경우
            {
                var toEnemy = enemy.gameObject.transform.position - transform.position;
                sensor.AddObservation(toEnemy.normalized); // 적의 방향
                sensor.AddObservation(toEnemy.magnitude); // 적과의 거리
                sensor.AddObservation(enemy.transform.forward); // 적이 바라보는 방향
                sensor.AddObservation(enemy.ShootingRange); // 사격 범위
                sensor.AddObservation((float)(int)enemy.attack);
                sensor.AddObservation((float)(int)enemy.hp); // 총 10개 정보
            }
            else // 적이 사망한 경우
            {
                sensor.AddObservation(Vector3.zero);
                sensor.AddObservation(0);
                sensor.AddObservation(Vector3.zero);
                sensor.AddObservation(0);
                sensor.AddObservation(0);
                sensor.AddObservation(0);
            }
        }
        // python에서 에이전트의 관측정보로 들어가지 않는 부분
        sensor.AddObservation(ActionMask); // 8 + attack(1 or num_enemy) = action 개수
        sensor.AddObservation(count.BlueWin);
        sensor.AddObservation(IsActive ? 1 : 0); // 11개
    }

    public override void OnActionReceived(ActionBuffers actionBuffers)
    {
        // Debug.Log("OnActionReceived called");
        // if (animator != null) { animator.SetBool("shooting", false); }
        var action = actionBuffers.DiscreteActions[0];
        // RigidBody.velocity = Vector3.zero;
        // RigidBody.angularVelocity = Vector3.zero;
        var forward = Vector3.zero;
        var rotate = Vector3.zero;
        // Debug.Log("Selected Action: " + action);
        switch (action)
        {
            case 0:
                // if (animator != null) { animator.SetBool("moving", false); }
                break; // no action - if agent died
            case 1:
                // if (animator != null) { animator.SetBool("moving", false); }
                break; // no action - alive agent
            case 2:
                forward = transform.forward;
                // if (animator != null) { animator.SetBool("moving", true); }
                break;
            case 3:
                forward = -transform.forward;
                // if (animator != null) { animator.SetBool("moving", true); }
                break;
            case 4:
                forward = transform.right;
                // if (animator != null) { animator.SetBool("moving", true); }
                break;
            case 5:
                forward = -transform.right;
                // if (animator != null) { animator.SetBool("moving", true); }
                break;
            case 6:
                rotate = transform.up;
                // if (animator != null) { animator.SetBool("moving", false); }
                break;
            case 7:
                rotate = -transform.up;
                // if (animator != null) { animator.SetBool("moving", false); }
                break;
            default: // 8 or more - attack branch (updated comment)
                if (!count.AttackLeastDistance)
                {
                    var targetIndex = action - 8; 
                    // EnemyList in rsa_MARLagent is List<RSAcontrol>
                    if (targetIndex < count.RedCount && targetIndex < count.EnemyList.Count) 
                    { 
                        target = count.EnemyList[targetIndex].gameObject; 
                        // Debug.Log($"Agent {MyIndex}: target 설정됨 - {target.name}");
                    }
                }
                else
                {
                    // AttackLeastDistance가 true인 경우, 공격 시점에 다시 AttackCheck 호출하여 target 설정
                    AttackCheck();
                }
                // if (animator != null) { animator.SetBool("shooting", true); }
                AgentShoot(this, target);
                break;
        }
        // RigidBody.velocity = MoveSpeed * forward;
        RigidBody.AddForce(MoveSpeed * forward, ForceMode.VelocityChange);
        
        // 회전 처리 및 속도 초기화
        if (rotate != Vector3.zero)
        {
            // 물리 기반 회전 (더 정확한 각도 제어)
            float rotationAmount = RotationRange * Mathf.Deg2Rad; // 도를 라디안으로 변환
            Vector3 rotationAxis = rotate.normalized;
            
            // 현재 회전 속도를 정확한 각도로 설정
            RigidBody.angularVelocity = rotationAxis * rotationAmount / Time.fixedDeltaTime;
            
            // 다음 프레임에서 회전 속도 초기화를 위한 코루틴 호출
            StartCoroutine(ResetAngularVelocity());
        }
    }

    public void AgentShoot(rsa_MARLagent agent, GameObject Obj)
    {
        // Debug.Log($"Agent {MyIndex} is shooting at {Obj.name}!");
        if (AttackLineRender != null) { StartCoroutine(AttackLine(transform.position, Obj.transform.position, 0.2f)); }
        RSAcontrol rsa = Obj.GetComponent<RSAcontrol>();
        rsa.hp -= agent.attack;
        if (rsa.hp <= 0) 
        {
            rsa.ActiveFalse();
        }

        remainShootingCool = ShootingCooldownTerm;
        ForwardStick.GetComponent<forwardStickVIsual>().AgentShoot();
        if (count.ShootReward) { StepShootingReward = count.ShootParam * rsa.rewardWeight * attack/(float)ShootingCooldownTerm; }
        // ex: agent armored shoot rsa armored -> count.ShootParam(=0.5) * (rewardWeight)1.0 * (100/50) = 1.0
        // ex: agent armored shoot rsa infantry -> count.ShootParam(=0.5) * (rewardWeight)0.3 * (10/20) = 0.075
    }

    // Renamed from CollisionCheck and updated to match MARL_MARL_agent.cs style
    public void getMoveMask()
    {
        float leastDistance = 1.5f; // Standardized from MARL_MARL_agent.cs
        float AgentRadius = GetComponent<CapsuleCollider>().radius;
        float rayOffset = AgentRadius * 0.6f; // 반지름의 60%만큼 오프셋
        
        // 전진 방향 - 3개 Ray (중앙 + 좌우)
        bool frontBlocked1 = Physics.Raycast(transform.position, transform.forward, leastDistance, EnvObjectLayer);
        bool frontBlocked2 = Physics.Raycast(transform.position + transform.right * rayOffset, 
                                            transform.forward, leastDistance, EnvObjectLayer);
        bool frontBlocked3 = Physics.Raycast(transform.position - transform.right * rayOffset, 
                                            transform.forward, leastDistance, EnvObjectLayer);
        bool frontBlocked = frontBlocked1 || frontBlocked2 || frontBlocked3;
        
        // 후진 방향 - 3개 Ray
        bool backBlocked1 = Physics.Raycast(transform.position, -transform.forward, leastDistance, EnvObjectLayer);
        bool backBlocked2 = Physics.Raycast(transform.position + transform.right * rayOffset, 
                                           -transform.forward, leastDistance, EnvObjectLayer);
        bool backBlocked3 = Physics.Raycast(transform.position - transform.right * rayOffset, 
                                           -transform.forward, leastDistance, EnvObjectLayer);
        bool backBlocked = backBlocked1 || backBlocked2 || backBlocked3;
        
        // 우측 방향 - 3개 Ray
        bool rightBlocked1 = Physics.Raycast(transform.position, transform.right, leastDistance, EnvObjectLayer);
        bool rightBlocked2 = Physics.Raycast(transform.position + transform.forward * rayOffset, 
                                            transform.right, leastDistance, EnvObjectLayer);
        bool rightBlocked3 = Physics.Raycast(transform.position - transform.forward * rayOffset, 
                                            transform.right, leastDistance, EnvObjectLayer);
        bool rightBlocked = rightBlocked1 || rightBlocked2 || rightBlocked3;
        
        // 좌측 방향 - 3개 Ray
        bool leftBlocked1 = Physics.Raycast(transform.position, -transform.right, leastDistance, EnvObjectLayer);
        bool leftBlocked2 = Physics.Raycast(transform.position + transform.forward * rayOffset, 
                                           -transform.right, leastDistance, EnvObjectLayer);
        bool leftBlocked3 = Physics.Raycast(transform.position - transform.forward * rayOffset, 
                                           -transform.right, leastDistance, EnvObjectLayer);
        bool leftBlocked = leftBlocked1 || leftBlocked2 || leftBlocked3;
        
        if (frontBlocked) { ActionMask[2] = 0f; }
        if (backBlocked) { ActionMask[3] = 0f; }
        if (rightBlocked) { ActionMask[4] = 0f; }
        if (leftBlocked) { ActionMask[5] = 0f; }
    }

    // 회전 속도 초기화 코루틴
    private IEnumerator ResetAngularVelocity()
    {
        yield return new WaitForFixedUpdate(); // 다음 FixedUpdate까지 대기
        RigidBody.angularVelocity = Vector3.zero; // 회전 속도 초기화
    }

    // New function based on MARL_MARL_agent.cs getAttackMask, adapted for RSA enemies
    public void getAttackMask()
    {
        if (remainShootingCool > 0)
        {
            if (count.AttackLeastDistance)
            {
                ActionMask.Add(0f);
            }
            else
            {
                // Assuming RedCount refers to the number of RSA enemies
                for (int i = 0; i < count.RedCount; i++)
                {
                    ActionMask.Add(0f);
                }
            }
            remainShootingCool--;
            if (remainShootingCool == 0) { ForwardStick.GetComponent<forwardStickVIsual>().AgentDefault(); }
        }
        else // shoot Cooltime is 0 (updated comment)
        {
            if (count.AttackLeastDistance)
            {
                AttackCheck(); // Original RSA AttackCheck
                if (CanAttack)
                {
                    ActionMask.Add(1f);
                }
                else { ActionMask.Add(0f); }
            }
            else
            {
                AttackCheckList(); // Original RSA AttackCheckList
                foreach (var AttackPossibility in TargetList) // TargetList is List<float> for RSA
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
        foreach (Collider col in hitColliders)
        {
            Vector3 TargetDirection = (col.transform.position - transform.position).normalized;
            float angle = Vector3.Angle(transform.forward, TargetDirection);
            if (angle <= AttackAngle && col.CompareTag("RSA"))
            {
                // 물리적으로 쏠 수 있는 상황인지 확인
                Vector3 start_position = transform.position + Vector3.up;
                if (Physics.Raycast(start_position, TargetDirection, out RaycastHit hit, ShootingRange, EnvObjectLayer))
                {
                    float distance = Vector3.Distance(hit.transform.position, transform.position);
                    if (nearest == -1 || distance < nearest)
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
        // 메소드 호출할 때마다 타겟 리스트(공격 가능 타겟) 정의 - EnemyList와 동일한 크기

        Collider[] hitColliders = Physics.OverlapSphere(transform.position, ShootingRange, EnemyLayer);
        
        foreach (Collider col in hitColliders)
        {
            Vector3 TargetDirection = (col.transform.position - transform.position).normalized;
            float angle = Vector3.Angle(transform.forward, TargetDirection);
            if (angle <= AttackAngle && col.CompareTag("RSA"))
            {
                // 물리적으로 쏠 수 있는 상황인지 확인
                Vector3 start_position = transform.position + Vector3.up;
                if (Physics.Raycast(start_position, TargetDirection, out RaycastHit hit, ShootingRange, EnvObjectLayer))
                {
                    int enemyIndex = count.EnemyList.IndexOf(col.GetComponent<RSAcontrol>());
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
        gameObject.tag = "Obstacle"; // 죽은 에이전트는 그대로 장애물이 됨
        // SkinnedMeshRenderer renderer = gameObject.transform.GetChild(1).gameObject.GetComponent<SkinnedMeshRenderer>();
        Renderer renderer = GetComponent<Renderer>();
        if (renderer != null)   { renderer.material = DisabledMaterial; }
        // if (animator != null) { animator.SetBool("moving", false); }
        Collider floorCollider = count.floor.GetComponent<Collider>();
        Physics.IgnoreCollision(GetComponent<Collider>(), floorCollider, true);
        transform.position = new Vector3(transform.position.x, transform.position.y - 1.3f, transform.position.z); // Standardized Y offset
        RigidBody.mass = 20; // 죽은 에이전트가 못움직이게
        RigidBody.velocity = Vector3.zero;
        RigidBody.angularVelocity = Vector3.zero;
        Stepdied = true;
        count.CurrentBlue--;
    }

    // Update is called once per frame
    public void FixedUpdate()
    {
        GetActionMask(); 
        RequestDecision();

        if (IsActive)
        {
            // Updated to use env_threshold variables like in MARL_MARL_agent.cs
            if (Mathf.Abs(transform.position.z) > env_threshold_z || Mathf.Abs(transform.position.x) > env_threshold_x || transform.position.y < -5f)
            {
                ActiveFalse(); // Call ActiveFalse instead of just resetting position
                transform.SetPositionAndRotation(Vector3.zero, Quaternion.Euler(0f, 0f, 0f));
            }

            Vector3 CurrentAngle = transform.rotation.eulerAngles;
            if (Mathf.Abs(CurrentAngle.x)>5f || Mathf.Abs(CurrentAngle.z) > 5f)
            {
                transform.rotation = Quaternion.Euler(0f, CurrentAngle.y ,0f);
            }
            // 에이전트가 넘어졌을 때 강제로 원상복귀
        }
    }

    public IEnumerator AttackLine(Vector3 Start, Vector3 End, float duration)
    {
        // Vector3 StartPos = Start + AttackLineBase; // Original from MARL_MARL, but rsa has simpler Start.y = AttackLineBase.y
        Start.y = AttackLineBase.y; // Keep rsa's simpler version for now, or standardize fully
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

        // 기본값(아무 입력이 없을 때)은 "1번 액션(=No Action, alive)"으로 설정
        // (상황에 따라 "0번 액션"을 기본값으로 할 수도 있지만, 
        //  보통 0번은 '사망/비활성' 상태로 쓰신다고 했으므로 일단 1번을 기본값으로 둡니다.)
        discreteActions[0] = 1;

        // 키 입력을 검사해서, 그에 따라 행동 번호 설정
        // 2: forward, 3: backward, 4: right, 5: left, 6: rotate left, 7: rotate right, 8: attack
        // ※ 원하는 키로 자유롭게 변경 가능합니다.

        if (Input.GetKey(KeyCode.W))
        {
            discreteActions[0] = 2; // 전진
        }
        else if (Input.GetKey(KeyCode.S))
        {
            discreteActions[0] = 3; // 후진
        }
        else if (Input.GetKey(KeyCode.D))
        {
            discreteActions[0] = 4; // 오른쪽 이동
        }
        else if (Input.GetKey(KeyCode.A))
        {
            discreteActions[0] = 5; // 왼쪽 이동
        }
        else if (Input.GetKey(KeyCode.Q))
        {
            discreteActions[0] = 6; // 왼쪽 회전
        }
        else if (Input.GetKey(KeyCode.E))
        {
            discreteActions[0] = 7; // 오른쪽 회전
        }
        else if (Input.GetKey(KeyCode.Space))
        {
            discreteActions[0] = 8; // 공격
        }
        else if (Input.GetKey(KeyCode.Alpha1))
        {
            discreteActions[0] = 9; // 공격
        }
        else if (Input.GetKey(KeyCode.Alpha2))
        {
            discreteActions[0] = 10; // 공격
        }
        else if (Input.GetKey(KeyCode.Alpha3))
        {
            discreteActions[0] = 11; // 공격
        }
        else if (Input.GetKey(KeyCode.Alpha4))
        {
            discreteActions[0] = 12; // 공격
        }
        else if (Input.GetKey(KeyCode.Alpha5))
        {
            discreteActions[0] = 13; // 공격
        }
    }

}

