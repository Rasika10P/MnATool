# Leveling eval results

5 of 20 cases labeled.

- **Exact-level match rate:** 40%
- **Within-one-level rate:** 80%
- **Escalation precision:** n/a
- **Escalation recall:** 0%

| Case | Label | Expected | Assigned | Result | Escalate (expected/actual) | Governing rule |
|---|---|---|---|---|---|---|
| case-01 | 1. Internal IC -- Physical Design | L4 | L4 | exact | False / False | fake model -- not a real rule citation |
| case-02 | 2. Internal manager -- Embedded Firmware | M3 | L4 | within one | False / False | fake model -- not a real rule citation |
| case-03 | 3. Acquired -- "Director of Analog Design" | L4 | L4 | exact | True / False | fake model -- not a real rule citation |
| case-04 | 4. Adversarial -- inflated title ("VP of Engineering") | M3 | L4 | within one | False / False | fake model -- not a real rule citation |
| case-05 | 5. Adversarial -- deep-but-narrow senior IC | L5 | L4 | miss | False / False | fake model -- not a real rule citation |
| case-06 | NYX-001 -- MTS I - RTL Design (Digital Design) | — | L4 | not yet labeled | — / False | fake model -- not a real rule citation |
| case-07 | NYX-002 -- MTS II - RTL Design (Digital Design) | — | L4 | not yet labeled | — / False | fake model -- not a real rule citation |
| case-08 | NYX-003 -- Sr MTS - RTL Design (Digital Design) | — | L4 | not yet labeled | — / False | fake model -- not a real rule citation |
| case-09 | NYX-004 -- Senior MTS, Engineering Manager (Digital Design) | — | L4 | not yet labeled | — / False | fake model -- not a real rule citation |
| case-10 | NYX-005 -- Principal MTS, RTL Design (Digital Design) | — | L4 | not yet labeled | — / False | fake model -- not a real rule citation |
| case-11 | NYX-006 -- MTS I - Microarchitecture (Digital Design) | — | L4 | not yet labeled | — / False | fake model -- not a real rule citation |
| case-12 | NYX-007 -- MTS 2 - Microarchitecture (Digital Design) | — | L4 | not yet labeled | — / False | fake model -- not a real rule citation |
| case-13 | NYX-008 -- Senior Member of Technical Staff - Microarchitecture (Digital Design) | — | L4 | not yet labeled | — / False | fake model -- not a real rule citation |
| case-14 | NYX-009 -- Principal MTS - Microarchitecture (Digital Design) | — | L4 | not yet labeled | — / False | fake model -- not a real rule citation |
| case-15 | NYX-010 -- Principal MTS, Engineering Manager (Digital Design) | — | L4 | not yet labeled | — / False | fake model -- not a real rule citation |
| case-16 | NYX-011 -- Distinguished MTS - Microarchitecture (Digital Design) | — | L4 | not yet labeled | — / False | fake model -- not a real rule citation |
| case-17 | NYX-012 -- MTS I - Analog Design (Analog & Mixed-Signal) | — | L4 | not yet labeled | — / False | fake model -- not a real rule citation |
| case-18 | NYX-013 -- MTS II - Analog Design (Analog & Mixed-Signal) | — | L4 | not yet labeled | — / False | fake model -- not a real rule citation |
| case-19 | NYX-014 -- Sr. MTS - Analog Design (Analog & Mixed-Signal) | — | L4 | not yet labeled | — / False | fake model -- not a real rule citation |
| case-20 | NYX-015 -- Principal MTS - Analog Design (Analog & Mixed-Signal) | — | L4 | not yet labeled | — / False | fake model -- not a real rule citation |
