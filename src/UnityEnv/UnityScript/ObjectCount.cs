using JetBrains.Annotations;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class ObjectCount : MonoBehaviour
{
    public bool RSAmode;
    public bool SparseReward;
    public bool killreward;
    public bool ShootReward;
    public float ShootParam=0.5f;
    public bool AttackLeastDistance = true;
    public GameObject floor;
    public List<rsa_MARLagent> AgentList = new();
    public List<RSAcontrol> EnemyList = new();

    public List<MARL_MARLagent> BlueList = new();
    public List<MARL_MARLagent> RedList = new();
    public int CurrentBlue;
    public int CurrentRed;
    public int BlueCount = 0;
    public int RedCount = 0;
    [HideInInspector] public int BlueWin;
    [HideInInspector] public int RedWin;
    // marl vs rsa
    [HideInInspector] public bool killedRSA = false;
    [HideInInspector] public float killedRSAreward;
    [HideInInspector] public bool DiedAgent = false;
    [HideInInspector] public float DiedAgentPenalty;
    // marl vs marl


    // Start is called before the first frame update
    void Start()
    {
        if (RSAmode)
        {
            BlueCount = AgentList.Count;
            RedCount = EnemyList.Count;
        }
        else
        {
            BlueCount = BlueList.Count;
            RedCount = RedList.Count;
        }
    }

    // Update is called once per frame
    void Update()
    {
        
    }
}
