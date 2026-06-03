"""
Modüler Ödül/Ceza Sistemi
=========================
Yeni bir ceza eklemek için:
  1. RewardConfig sınıfına yeni parametre ekle
  2. compute_reward() fonksiyonuna hesaplama bloğu ekle
  3. Hepsi bu kadar!
"""
from dataclasses import dataclass


@dataclass
class RewardConfig:
    """Her bir cezanın ağırlığını buradan ayarla."""

    # Terminal penalties
    out_of_road_penalty: float = -20.0
    crash_vehicle_penalty: float = -50.0    # raised from -30: crashing must clearly cost more than
                                            # briefly slowing/stopping, or the policy "drives through" lead cars
    crash_object_penalty: float = -20.0

    # Terminal rewards
    arrive_dest_bonus: float = 50.0

    # Continuous rewards / penalties
    route_progress_scale: float = 40.0      # raised 10->40: completing the route must DOMINATE the reward.
                                            # At 10, 80% vs 50% route differed by only +3, so the policy had no
                                            # incentive to finish and plateaued at "drive-around-then-fail" (~+20).
                                            # At 40, full route = +40 (vs ~+30 dense), so finishing clearly wins.
                                            # Spread per-step (route_delta x 40), so it stays under the +-10 clip.
    harsh_steering_threshold: float = 0.3   # |steer| below this is normal cornering, not penalized
    harsh_steering_weight: float = -0.1     # applied to excess above threshold, speed-scaled
    steering_diff_penalty: float = -0.03    # raised from -0.01: directly penalises the steering wobble (jerk), speed-scaled
    speed_scale_ref: float = 50.0           # km/h reference for speed-scaling steering penalties
    speed_bonus_weight: float = 0.1         # reduced from 0.3; was too strong and encouraged reckless speed
    speed_bonus_min: float = 3.0            # no bonus below this speed (km/h)
    standing_still_penalty: float = -0.03   # softened from -0.1: braking for a lead car is legitimate and must
                                            # not be punished harder than crashing; speed_bonus still rewards driving when clear

    # Heading alignment — fires every step, dense signal even when route_progress ≈ 0
    heading_alignment_weight: float = 0.03  # max reward per step when perfectly aligned with road
    heading_alignment_max_diff: float = 0.5 # radians — beyond this the reward is zero

    # Speed limit
    speed_limit: float = 60.0              # km/h — raised to match IL model's trained operating speed
    speed_limit_weight: float = -0.05      # per km/h over the limit; -0.2 was drowning route progress signal

    # Lane-keeping
    lateral_offset_threshold: float = 0.5   # metres from lane centre — free zone, no penalty
    lateral_offset_weight: float = -0.3     # penalty per metre beyond threshold, per step
    # Passing suppression: lateral penalty fades linearly to 0 as |steer| reaches this value.
    # Deliberate lateral manoeuvres (passing, avoidance) require nonzero steering, so the
    # penalty is suppressed when the agent is actively steering and fully applied when drifting
    # passively (steer ≈ 0).
    lateral_suppression_steer: float = 0.15  # |steer| at which penalty is fully suppressed

    # Forward proximity — the DENSE braking signal. Terminal crash penalty is clipped
    # to -10 (see env_wrapper), so it can't teach braking on its own; this per-step
    # penalty accumulates as the car approaches a lead vehicle, giving a gradient to
    # slow down BEFORE the crash. Scaled by speed, so charging a close car is worst and
    # braking (which drops speed) directly reduces it. ~0 when the road ahead is clear.
    front_safe_dist: float = 25.0            # metres; start penalising a lead car closer than this
                                             # (raised 15->25: with decision_repeat=5 the car needs a
                                             #  longer runway to brake before it's too late)
    front_proximity_weight: float = -1.2     # per-step weight, x closeness(0..1) x speed_factor(0..1)
                                             # (raised -0.6->-1.2 so braking clearly beats the +0.1/step
                                             #  speed bonus + route-progress reward for plowing forward)


def compute_reward(info: dict,
                   action,
                   prev_route_completion: float,
                   speed: float,
                   cfg: RewardConfig = None,
                   prev_action=None,
                   heading_diff: float = 0.0,
                   front_dist: float = 100.0) -> tuple[float, dict]:
    """
    Toplam ödülü ve detaylı döküm sözlüğünü döndürür.

    Args:
        info:  MetaDrive env.step() çıktısı
        action: [steering, throttle] numpy array
        prev_route_completion: önceki adımdaki rota tamamlanma oranı
        speed: araç hızı (km/h)
        cfg: ödül ayarları (None ise varsayılan kullanılır)

    Returns:
        (toplam_ödül, {"ceza_adı": değer, ...})
    """
    if cfg is None:
        cfg = RewardConfig()

    reward = 0.0
    details = {}

    # 1. Route progress
    route_delta = info.get("route_completion", 0.0) - prev_route_completion
    r = route_delta * cfg.route_progress_scale
    reward += r
    details["route_progress"] = r

    # 2. Out of road
    if info.get("out_of_road", False):
        reward += cfg.out_of_road_penalty
        details["out_of_road"] = cfg.out_of_road_penalty

    # 3. Crash vehicle
    if info.get("crash_vehicle", False):
        reward += cfg.crash_vehicle_penalty
        details["crash_vehicle"] = cfg.crash_vehicle_penalty

    # 4. Crash object
    if info.get("crash_object", False):
        reward += cfg.crash_object_penalty
        details["crash_object"] = cfg.crash_object_penalty

    # 5. Arrive destination
    if info.get("arrive_dest", False):
        reward += cfg.arrive_dest_bonus
        details["arrive_dest"] = cfg.arrive_dest_bonus

    # 6. Harsh steering — only fires above threshold, scaled by speed so low-speed turns are free
    speed_factor = min(speed, cfg.speed_scale_ref) / cfg.speed_scale_ref
    steer_val = abs(float(action[0]))
    excess = max(0.0, steer_val - cfg.harsh_steering_threshold)
    steer_pen = cfg.harsh_steering_weight * excess * speed_factor
    reward += steer_pen
    details["harsh_steering"] = steer_pen

    # 7. Steering jerk — speed-scaled so high-speed jerks are penalized more
    if prev_action is not None:
        steer_diff = abs(float(action[0]) - float(prev_action[0]))
        diff_pen = cfg.steering_diff_penalty * steer_diff * speed_factor
        reward += diff_pen
        details["steering_diff"] = diff_pen

    # 8. Speed bonus / standing still penalty
    if speed > cfg.speed_bonus_min:
        bonus = cfg.speed_bonus_weight * min(speed, cfg.speed_scale_ref) / cfg.speed_scale_ref
        reward += bonus
        details["speed_bonus"] = bonus
    else:
        reward += cfg.standing_still_penalty
        details["standing_still"] = cfg.standing_still_penalty

    # 9. Lateral offset — penalise passive drift out of lane, not intentional passing manoeuvres.
    # Suppression logic: a deliberate lateral movement (passing, avoidance) requires the agent
    # to actively steer. Passive drift has |steer| ≈ 0. The penalty therefore scales with
    # (1 - |steer| / suppression_steer), clamped to [0, 1], so:
    #   |steer| = 0          → full penalty  (drifting)
    #   |steer| = 0.075      → half penalty
    #   |steer| ≥ 0.15       → no penalty    (active passing/avoidance)
    lateral_dist = info.get("lateral_dist", 0.0)
    excess_lateral = max(0.0, abs(lateral_dist) - cfg.lateral_offset_threshold)
    suppression = max(0.0, 1.0 - steer_val / cfg.lateral_suppression_steer)
    lat_pen = cfg.lateral_offset_weight * excess_lateral * suppression
    reward += lat_pen
    details["lateral_offset"] = lat_pen

    # 10. Speed limit — penalise every km/h over the limit
    overspeed = max(0.0, speed - cfg.speed_limit)
    speed_limit_pen = cfg.speed_limit_weight * overspeed
    reward += speed_limit_pen
    details["speed_limit"] = speed_limit_pen

    # 11. Heading alignment — dense signal for staying pointed along the road
    align_reward = cfg.heading_alignment_weight * max(0.0, 1.0 - abs(heading_diff) / cfg.heading_alignment_max_diff)
    reward += align_reward
    details["heading_alignment"] = align_reward

    # 12. Forward proximity — DENSE braking signal (the terminal crash penalty is
    # clipped to -10, so it can't teach braking by itself). Penalty grows as the car
    # gets closer to a lead vehicle and as it goes faster, so the gradient points to
    # slow down before the crash; braking lowers speed_factor and directly relieves it.
    if front_dist < cfg.front_safe_dist:
        closeness = (cfg.front_safe_dist - front_dist) / cfg.front_safe_dist   # 0..1
        prox_pen  = cfg.front_proximity_weight * closeness * speed_factor
        reward   += prox_pen
        details["front_proximity"] = prox_pen

    # ──────────────────────────────────────────────
    #  YENİ CEZA EKLEMEK İÇİN BURAYA YAZ
    # ──────────────────────────────────────────────

    return reward, details
