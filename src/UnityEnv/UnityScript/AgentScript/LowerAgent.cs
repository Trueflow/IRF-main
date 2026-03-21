using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Policies;
using Unity.MLAgents.Sensors;

public class LowerAgent : Agent
{
    // RSA �ó����� ������ ����մϴ�.가나다라
    
    public enum Position
    {
        Infantry = 0, //����
        Armored = 1, // �Ⱙ - �� ������
        Tank = 2 // ���� - �� ����
    }
    public HRL_MARL_controller EnvController;
    [HideInInspector] public int MyIndex;

    [HideInInspector] public BehaviorParameters Behavior;
    [HideInInspector] public Rigidbody RigidBody;
    public Position position;

    // ������Ʈ�� Ȱ��ȭ ���θ� �ð������� ǥ���ϱ� ���� �뵵�� �߰��߽��ϴ�.
    public Material DefaultMaterial;
    public Material DisabledMaterial;

    // ������Ʈ�� �Ӽ��� ���õ� �����Դϴ�.
    [HideInInspector] float MoveSpeed;
    [HideInInspector] float RotationRange;
    [HideInInspector] float ShootingRange;
    [HideInInspector] public float rewardWeight = 1f;

    [HideInInspector] public float attack;
    [HideInInspector] public float hpMax;
    [HideInInspector] public float hp;

    [HideInInspector] public int ShootingTerm = 0;

    public LayerMask RaycastLayer;

    [HideInInspector] public Vector3 Direction;
    [HideInInspector] public float Distance;

    public override void Initialize()
    {
        RigidBody = GetComponent<Rigidbody>();
        switch (position)
        {
            case (Position)0: // ����
                MoveSpeed = 0.8f;
                RotationRange = 60f;
                ShootingRange = 5f;
                rewardWeight = 1f;
                attack = 30f;
                hpMax = 100f;

                break;
            case (Position)1: // �Ⱙ
                MoveSpeed = 0.25f;
                RotationRange = 25f;
                ShootingRange = 8f;
                rewardWeight = 3f;
                attack = 60f;
                hpMax = 300f;
                break;
            case (Position)2: // ��ũ
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
        // UpperAgent�� �ִ� ����
        sensor.AddObservation(hp);
        sensor.AddObservation(EnvController.LowerAgentList[MyIndex].IsActive ? 1 : 0);
    }
    public override void WriteDiscreteActionMask(IDiscreteActionMask actionMask)
    {
        // OnActionRecieved ���� ����Ǵ� �޼ҵ��Դϴ�.
        // ������Ʈ�� �� �� ���� �ൿ�� ��Ȱ��ȭ ��Ű�� �ܰ��Դϴ�.
        if (!EnvController.LowerAgentList[MyIndex].IsActive)
        {
            // branch 0 : move
            actionMask.SetActionEnabled(0, 1, false);
            actionMask.SetActionEnabled(0, 2, false);
            actionMask.SetActionEnabled(0, 3, false);
            //actionMask.SetActionEnabled(0, 4, false);
        }
        /*        else
                {
                    actionMask.SetActionEnabled(0, 0, false);
                }*/
    }

    public override void OnActionReceived(ActionBuffers actionBuffers)
    {
        var move = actionBuffers.DiscreteActions[0];

        switch (move)
        {
            case 0:
                break; // no action
            case 1:
                RigidBody.AddForce(-1.4f * MoveSpeed * transform.forward, ForceMode.VelocityChange);
                break;
            case 2:
                RigidBody.AddForce(0.7f * MoveSpeed * transform.forward, ForceMode.VelocityChange);
                transform.Rotate(transform.up, RotationRange * Time.deltaTime);
                //transform.Rotate(rotateDir, 100f * Time.deltaTime);
                break;
            case 3:
                RigidBody.AddForce(0.7f * MoveSpeed * transform.forward, ForceMode.VelocityChange);
                transform.Rotate(transform.up, -RotationRange * Time.deltaTime);
                //transform.Rotate(rotateDir, -100f * Time.deltaTime);
                break;
        }
    }

    public void ActiveFalse()
    {
        //Debug.Log("Agent ActiveFalse called");
        EnvController.LowerAgentList[MyIndex].IsActive = false;
        gameObject.tag = "DiedObj";
        // SkinnedMeshRenderer renderer = gameObject.transform.GetChild(1).gameObject.GetComponent<SkinnedMeshRenderer>();
        Renderer renderer = GetComponent<Renderer>();
        if (renderer != null)
        {
            renderer.material = DisabledMaterial;
        }
        Collider floorCollider = EnvController.floor.GetComponent<Collider>();
        Physics.IgnoreCollision(GetComponent<Collider>(), floorCollider, true);
        transform.position = new Vector3(transform.position.x, transform.position.y - 0.6f, transform.position.z);
        RigidBody.mass = 20; // ���� ������Ʈ�� �������̰�
    }

    // Update is called once per frame
    public void FixedUpdate()
    {
        RequestDecision();
        if (ShootingTerm > 0)
        {
            ShootingTerm--;
        }
        if (EnvController.LowerAgentList[MyIndex].IsActive)
        {
            //Debug.DrawRay(transform.position, transform.forward * ShootingRange, Color.blue);
            if (Physics.Raycast(transform.position, transform.forward, out RaycastHit hit, ShootingRange, RaycastLayer) && ShootingTerm == 0)
            {
                //Debug.DrawRay(transform.position, transform.forward * ShootingRange, Color.red);
                EnvController.LowerShoot(this, hit.collider.gameObject);
                ShootingTerm = 5;
            }
            if (Mathf.Abs(transform.position.z) > 10f | Mathf.Abs(transform.position.x) > 10f)
            {
                ActiveFalse();
                EnvController.CurrentLower--;
                transform.SetPositionAndRotation(Vector3.zero, Quaternion.Euler(0f, 0f, 0f));
            }
        }
    }
}
