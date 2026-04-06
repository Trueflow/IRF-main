using System.Collections;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;
using System;

public class RSAcontrol : MonoBehaviour
{
    public enum Position
    {
        Infantry = 0, // Infantry
        Armored = 1,  // Armored
        Artillery = 2 // Artillery (matching rsa_MARLagent's enum after user change)
    }
    public ObjectCount count;
    // public RSA_marl_Controller EnvController;

    [HideInInspector] public int MyIndex;

    [HideInInspector] public Rigidbody RigidBody;
    public Position position;
    // 에이전트의 활성화 여부를 시각적으로 표시하기 위한 용도로 추가했습니다.
    public Material DefaultMaterial;
    public Material DisabledMaterial;
    public GameObject ForwardStick;
    // 에이전트의 속성에 관련된 변수입니다.
    [HideInInspector] float MoveSpeed;
    [HideInInspector] float RotationRange;
    [HideInInspector] public float ShootingRange;
    [HideInInspector] public float rewardWeight;

    [HideInInspector] public float attack;
    [HideInInspector] public float hp;
    public float hpMax;
    [HideInInspector] public float InitialY;

    [HideInInspector] public int ShootingCooldownTerm = 30;
    [HideInInspector] public int remainShootingCool = 0;
    public float AttackAngle = 20f;

    [HideInInspector] public bool IsActive = true;
    [HideInInspector] public List<float> ActionMask;
    [HideInInspector] public bool CanAttack;
    [HideInInspector] public GameObject target;

    public LayerMask EnemyLayer;
    public LayerMask EnvObjectLayer;
    public LineRenderer AttackLineRender;
    [HideInInspector] public Vector3 AttackLineBase = new Vector3(0f, -1f, 0f);

    [HideInInspector] public float StepShootingReward;
    public List<float> TargetList;
    [HideInInspector] public bool Stepdied = false;

    [HideInInspector] public float env_threshold_x;
    [HideInInspector] public float env_threshold_z;

    void Awake()
    {
        if (AttackLineRender != null) { AttackLineRender.enabled = false; }
        RigidBody = GetComponent<Rigidbody>();
        if (count.SparseReward)
        {
            switch (position)
            {
                case Position.Infantry: // Infantry
                    ShootingRange = 5f;
                    attack = 10f;
                    MoveSpeed = 1f;
                    RotationRange = 5f;
                    rewardWeight = 0.4f;
                    hpMax = 100f;
                    ShootingCooldownTerm = 5; 
                    InitialY = -1f;
                    break;
                case Position.Armored: // Armored
                    ShootingRange = 10f;
                    attack = 50f;
                    MoveSpeed = 2f;
                    RotationRange = 2f;
                    rewardWeight = 1f;
                    hpMax = 500f;
                    ShootingCooldownTerm = 20;
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
            }
        }
        else
        {
            switch (position)
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
            }
        }
        
        env_threshold_x = count.floor.transform.localScale.x / 2f;
        env_threshold_z = count.floor.transform.localScale.z / 2f;
    }
    
    public void Move()
    {
        List<int> availableIndices = ActionMask
            .Select((value, index) => new { value, index })
            .Where(x => x.value == 1f)
            .Select(x => x.index)
            .ToList();
        int action = availableIndices[UnityEngine.Random.Range(0, availableIndices.Count)];
        // Debug.Log(element);
        // RigidBody.velocity = Vector3.zero;
        // RigidBody.angularVelocity = Vector3.zero; // 회전 멈춤
        var forward = Vector3.zero;
        var rotate = Vector3.zero;
        switch (action)
        {
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
                rotate = -1f * transform.up;
                // if (animator != null) { animator.SetBool("moving", false); }
                break;
            default: // attack (action >=8),
                if (!count.AttackLeastDistance)
                {
                    // action 개수 ; default action 포함 이동/회전 8개 + 적 에이전트 수
                    var targetIndex = action - 8; // 사격 불가능한 적은 마스킹되어 타겟으로 지정될 일 없음
                    if (targetIndex < count.BlueCount && targetIndex < count.AgentList.Count) { target = count.AgentList[targetIndex].gameObject; }
                }
                // if (animator != null) { animator.SetBool("shooting", true); }
                if (target != null) { RSAShoot(this, target); }
                break;
        }
        // RigidBody.velocity = MoveSpeed * forward;
        RigidBody.AddForce(MoveSpeed * forward, ForceMode.VelocityChange);
        transform.Rotate(rotate, RotationRange);
        
    }

    public void RSAShoot(RSAcontrol rsa, GameObject Obj)
    {
        if (AttackLineRender != null) { StartCoroutine(AttackLine(transform.position, Obj.transform.position, 0.2f)); }
        rsa_MARLagent agent = Obj.GetComponent<rsa_MARLagent>();
        agent.hp -= rsa.attack;
        if (agent.hp <= 0)
        {
            agent.ActiveFalse();
        }
        remainShootingCool = ShootingCooldownTerm;
        ForwardStick.GetComponent<forwardStickVIsual>().AgentShoot();
    }

    // Update is called once per frame
    public void FixedUpdate()
    {
        if (IsActive)
        {
            GetActionMask();
            Move();
            // 환경 밖으로 이탈한 에이전트 비활성화
            if (Mathf.Abs(transform.position.z) > env_threshold_z || Mathf.Abs(transform.position.x) > env_threshold_x || transform.position.y < -5f)
            {
                ActiveFalse();
                transform.SetPositionAndRotation(Vector3.zero, Quaternion.Euler(0f, 0f, 0f));
            }
            // 에이전트가 넘어졌을 때 강제로 원상복귀
            Vector3 CurrentAngle = transform.rotation.eulerAngles;
            if (Mathf.Abs(CurrentAngle.x) > 5f || Mathf.Abs(CurrentAngle.z) > 5f)
            {
                transform.rotation = Quaternion.Euler(0f, CurrentAngle.y, 0f);
            }
            // 에이전트가 넘어졌을 때 강제로 원상복귀
        }
    }

    public void GetActionMask()
    {
        ActionMask = new() { 0f, 1f, 1f, 1f, 1f, 1f, 1f, 1f };
        getMoveMask();
        getAttackMask();
    }

    public void getMoveMask()
    {
        float leastDistance = 1.5f;
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
        if (remainShootingCool > 0)
        {
            if (count.AttackLeastDistance)
            {
                ActionMask.Add(0f);
            }
            else
            {
                for (int i = 0; i < count.BlueCount; i++)
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
        foreach (Collider col in hitColliders)
        {
            Vector3 TargetDirection = (col.transform.position - transform.position).normalized;
            float angle = Vector3.Angle(transform.forward, TargetDirection);
            if (angle <= AttackAngle && col.CompareTag("MARLagent"))
            {
                // 물리적으로 쏠 수 있는 상황인지 확인
                Vector3 start_position = transform.position + Vector3.up;
                // Debug.DrawRay(start_position, TargetDirection * ShootingRange, Color.blue);
                if (Physics.Raycast(start_position, TargetDirection, out RaycastHit hit, ShootingRange, EnvObjectLayer))
                {
                    float distance = Vector3.Distance(col.transform.position, transform.position);
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
        TargetList = Enumerable.Repeat(0f, count.BlueCount).ToList();
        // 메소드 호출할 때마다 타겟 리스트(공격 가능 타겟) 정의 - EnemyList와 동일한 크기

        Collider[] hitColliders = Physics.OverlapSphere(transform.position, ShootingRange, EnemyLayer);

        foreach (Collider col in hitColliders)
        {
            Vector3 TargetDirection = (col.transform.position - transform.position).normalized;
            float angle = Vector3.Angle(transform.forward, TargetDirection);
            if (angle <= AttackAngle && col.CompareTag("MARLagent"))
            {
                // 물리적으로 쏠 수 있는 상황인지 확인
                Vector3 start_position = transform.position + Vector3.up;
                if (Physics.Raycast(start_position, TargetDirection, out RaycastHit hit, ShootingRange, EnvObjectLayer))
                {
                    int enemyIndex = count.AgentList.IndexOf(col.GetComponent<rsa_MARLagent>());
                    if (enemyIndex != -1) { TargetList[enemyIndex] = 1f; }
                }
            }
        }
    }

    public void ActiveFalse()
    {
        if (!IsActive) return;
        hp = 0f;
        IsActive = false;
        gameObject.tag = "Obstacle";
        // SkinnedMeshRenderer renderer = gameObject.transform.GetChild(1).gameObject.GetComponent<SkinnedMeshRenderer>();
        Renderer renderer = GetComponent<Renderer>();
        if (renderer != null)   { renderer.material = DisabledMaterial; }
        // if (animator != null) { animator.SetBool("moving", false); }
        Collider floorCollider = count.floor.GetComponent<Collider>();
        if (floorCollider != null && GetComponent<Collider>() != null)
        {
             Physics.IgnoreCollision(GetComponent<Collider>(), floorCollider, true);
        }
        transform.position = new Vector3(transform.position.x, transform.position.y - 1.3f, transform.position.z);
        if (RigidBody != null)
        {
            RigidBody.mass = 10; // 죽은 에이전트가 못움직이게
            RigidBody.velocity = Vector3.zero;
            RigidBody.angularVelocity = Vector3.zero;
        }
        Stepdied = true;
        count.CurrentRed--;
    }

    public IEnumerator AttackLine(Vector3 Start, Vector3 End, float duration)
    {
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
}
