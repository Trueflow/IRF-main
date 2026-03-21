using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Policies;
using Unity.MLAgents.Sensors;
using System;
using System.Linq;

public class DefaultAgent : Agent
{
        // RSA Scenario Agent Script (updated comment)
    public enum Position
    {
        Infantry = 0, // Infantry (updated comment)
        Armored = 1, // Armored (updated comment)
        Artillery = 2 // Artillery (updated comment, though original was Tank, MARL_MARL uses Artillery for 2)
    }
    public ObjectCount count;
    public RewardManager rewardManager;
    public float StepShootingReward;
    public List<float> TargetList; // This is List<float> in rsa, List<MARL_MARLagent> in MARL_MARL. Keeping List<float>.
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

    public LayerMask EnemyLayer;
    public LayerMask EnvObjectLayer;
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
    [HideInInspector] public bool Stepdied = false; // Standardized from MARL_MARL_agent.cs

    [HideInInspector] public float env_threshold_x; // Added from MARL_MARL_agent.cs
    [HideInInspector] public float env_threshold_z; // Added from MARL_MARL_agent.cs
    // Start is called before the first frame update
    void Start()
    {
        
    }

    // Update is called once per frame
    void Update()
    {
        
    }
}
