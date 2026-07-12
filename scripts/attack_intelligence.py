ATTACK_INTELLIGENCE = {
    "Normal": {
        "severity": "Low",
        "description": "This traffic is considered normal network behavior with no malicious activity detected.",
        "danger_level": "Safe",
        "performance_impact": "No harmful effect on system or network performance.",
        "system_effect": "Regular traffic flow without attack behavior.",
        "recommendation": "No action required. Continue normal monitoring."
    },
    "Analysis": {
        "severity": "Medium",
        "description": "Analysis attacks involve suspicious inspection or probing of services and system behavior.",
        "danger_level": "Moderate",
        "performance_impact": "Can increase small-scale traffic and system monitoring load.",
        "system_effect": "May indicate an attacker is studying the environment before launching a real attack.",
        "recommendation": "Monitor source activity, inspect logs, and block repeated suspicious behavior."
    },
    "Backdoor": {
        "severity": "High",
        "description": "A backdoor attack attempts to create unauthorized hidden access to a system.",
        "danger_level": "High Risk",
        "performance_impact": "May not immediately reduce performance, but can enable long-term compromise.",
        "system_effect": "Allows attackers to bypass normal authentication and gain persistent control.",
        "recommendation": "Isolate the affected system, inspect unauthorized services, and perform malware scanning."
    },
    "DoS": {
        "severity": "High",
        "description": "Denial of Service attacks overwhelm a target with traffic or requests to make services unavailable.",
        "danger_level": "High Risk",
        "performance_impact": "Causes heavy CPU, memory, and bandwidth usage; may slow down or crash services.",
        "system_effect": "Users may experience service downtime, slow response, or total unavailability.",
        "recommendation": "Apply rate limiting, firewall filtering, IP blocking, and traffic mitigation mechanisms."
    },
    "Exploits": {
        "severity": "High",
        "description": "Exploit attacks take advantage of software vulnerabilities to gain unauthorized access or control.",
        "danger_level": "High Risk",
        "performance_impact": "Can destabilize services and cause abnormal system behavior.",
        "system_effect": "May lead to privilege escalation, code execution, data theft, or system compromise.",
        "recommendation": "Patch vulnerable systems, review logs, isolate impacted hosts, and update security controls."
    },
    "Fuzzers": {
        "severity": "Medium",
        "description": "Fuzzer attacks send unexpected or malformed inputs to discover vulnerabilities in applications or protocols.",
        "danger_level": "Moderate",
        "performance_impact": "May increase CPU load and trigger crashes in weak services.",
        "system_effect": "Can expose software weaknesses and create instability.",
        "recommendation": "Validate inputs strictly, patch software, and monitor for repeated malformed requests."
    },
    "Generic": {
        "severity": "Medium",
        "description": "Generic attacks represent broad malicious patterns not restricted to one narrow exploit type.",
        "danger_level": "Moderate to High",
        "performance_impact": "May vary depending on the exact traffic behavior.",
        "system_effect": "Can indicate malicious activity targeting general weaknesses in the environment.",
        "recommendation": "Inspect related flows, review security logs, and strengthen perimeter monitoring."
    },
    "Reconnaissance": {
        "severity": "Medium",
        "description": "Reconnaissance attacks scan systems, ports, or services to collect information before future attacks.",
        "danger_level": "Moderate",
        "performance_impact": "Usually causes light to moderate network overhead.",
        "system_effect": "Helps attackers map the target environment and identify weak points.",
        "recommendation": "Block repeated scanners, enable IDS alerts, and reduce exposed services."
    },
    "Shellcode": {
        "severity": "High",
        "description": "Shellcode attacks involve malicious payloads intended to execute commands on the victim system.",
        "danger_level": "High Risk",
        "performance_impact": "Can trigger abnormal process execution and system instability.",
        "system_effect": "May lead to remote code execution and full compromise of the target machine.",
        "recommendation": "Immediately isolate the affected host, inspect running processes, and scan for malware."
    },
    "Worms": {
        "severity": "Critical",
        "description": "Worm attacks self-replicate and spread automatically across systems and networks.",
        "danger_level": "Critical",
        "performance_impact": "Can heavily consume CPU, memory, disk, and network bandwidth.",
        "system_effect": "Rapid spreading can infect multiple hosts and disrupt large parts of the network.",
        "recommendation": "Isolate infected systems immediately, block lateral movement, and start incident response procedures."
    }
}

def get_attack_intelligence(attack_name: str):
    return ATTACK_INTELLIGENCE.get(attack_name, {
        "severity": "Unknown",
        "description": "No detailed intelligence available for this attack type.",
        "danger_level": "Unknown",
        "performance_impact": "Unknown",
        "system_effect": "Unknown",
        "recommendation": "Review manually and inspect logs."
    })