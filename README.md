# EXOLAB Intern Design Optimization

하지 외골격 로봇의 관절 설계 최적화를 위한 연구 프로젝트.
인체 골격 모델(OpenSim 기반 URDF)을 활용하여, 무릎·엉덩이 관절의 기구학적 최적 설계를 수행한다.

---

## Repository Structure

```
human_robot_model/
├── myobody.urdf      # OpenSim Myosuit 기반 인체 skeletal URDF 모델
├── urdf_check.py     # URDF 파싱 및 관절/링크 정보 확인 스크립트
└── README.md
```

---

## Getting Started

### Prerequisites

```bash
pip install yourdfpy numpy scipy
```

### URDF 모델 확인

```bash
python urdf_check.py
```

`urdf_check.py`를 실행하면 `myobody.urdf`의 관절 구조, 링크 질량/관성, 관절 축 정보 등을 확인할 수 있다.

---

## Intern Research Guide

### Step 1: Knee Joint 분석

**목표:** 실제 사람의 무릎 관절(rotation + translation 동시 발생)을 단일 revolute joint로 근사할 때, kinematic deflection이 최소화되는 revolute joint 위치를 도출한다.

#### 문제 정의

- Thigh frame과 shank frame만 사용
- 실제 무릎은 flexion 시 rotation과 translation이 동시에 발생 (rolling + sliding)
- 이를 단일 revolute joint로 근사하면 필연적으로 kinematic deflection 발생
- **목표: deflection을 최소화하는 revolute joint 위치 (x, y, z) 탐색**

#### 수행 내용

1. **인체 모델 로드**: `myobody.urdf`에서 thigh, shank 링크 및 knee joint 정보 추출
2. **보행 데이터 활용**: knee flexion/extension 궤적(gait cycle) 기반 분석
3. **최적화**:
   - Grid search 또는 gradient-based optimization 적용
   - Revolute joint 위치를 변수로, deflection(위치 오차 norm)을 목적함수로 설정
4. **Internal load 추정**: 최적 위치에서 발생하는 force 및 moment 계산

#### Key Concepts

- Revolute joint 근사 시 발생하는 deflection = 실제 궤적과 근사 궤적의 차이
- Deflection이 클수록 외골격-인체 간 misalignment → 착용자에게 부하 전달

---

### Step 2: Hip 3축 Mechanism 설계 최적화

**목표:** Hip joint (3-DOF: flexion/extension, ab/adduction, internal/external rotation) 구현을 위한 mechanism 설계 최적화.

#### 문제 정의

> 3개 액추에이터를 어떻게 배치해야, 보행과 앉기/서기에 필요한 workspace를 전부 커버하면서, 각 액추에이터의 구동력이 현실적 범위 내에 있고, 인체와의 정합성도 확보되는가?

#### 설계 변수 및 제약

| 항목 | 설명 |
|------|------|
| **Torque profile** | 보행 및 sit-to-stand 시 hip joint torque |
| **Jacobian 변환** | 축 배치에 따라 hip torque → 각 액추에이터 필요 구동력 변환 |
| **Feasibility** | 구동력이 상용 액추에이터 스펙 초과 시 infeasible |
| **정합성** | 인체 해부학적 축과의 misalignment이 크면 관절 내부 부하 발생 |

#### 최적화 방법론

1. **RL 기반 최적화**: 축 배치를 action으로, workspace coverage + 구동력 feasibility + 정합성을 reward로 설정
2. **Iterative tuning**: 수렴 후 인체 정합성이 부족하면 reward weight 조정 후 재학습
3. **Validation**: 최적 설계안에 대해 보행/sit-to-stand 시나리오 시뮬레이션 검증

---

## References

- Winter, D.A. (2009). *Biomechanics and Motor Control of Human Movement* — Ch.1–3, 9
- OpenSim Gait2392 모델 문서: https://simtk-confluence.stanford.edu/display/OpenSim/Gait+2392+and+2354+Models
