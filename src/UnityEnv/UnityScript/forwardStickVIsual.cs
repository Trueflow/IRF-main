using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class forwardStickVIsual : MonoBehaviour
{
    public Material DefaultMaterial;
    public Material HitMaterial;
    public Material CooldownMaterial;
    // Start is called before the first frame update
    void Start()
    {
        Renderer renderer = GetComponent<Renderer>();
        if (renderer != null)
        {
            renderer.material = DefaultMaterial;
        }
    }

    // Update is called once per frame
    void Update()
    {
        
    }

    public void AgentShoot()
    {
        Renderer renderer = GetComponent<Renderer>();
        if (renderer != null) 
        { 
            //renderer.material = HitMaterial;
            renderer.material = CooldownMaterial;
        }
    }
    public void AgentDefault()
    {
        Renderer renderer = GetComponent<Renderer>();
        if (renderer != null)
        {
            renderer.material = DefaultMaterial;
        }
    }
}
