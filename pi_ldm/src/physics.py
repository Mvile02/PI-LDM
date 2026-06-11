import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Differentiable Atmospheric and Speed Conversion Formulas in PyTorch ---

def compute_atmos_torch(h_m, dT=0.0):
    """
    Computes pressure, density, and temperature at altitude (meters) in PyTorch.
    Differentiable approximation of OpenAP / ISA equations.
    """
    T0 = 288.15
    R = 287.05287
    rho0 = 1.225
    
    device = h_m.device if torch.is_tensor(h_m) else None
    
    if not torch.is_tensor(h_m):
        h_m = torch.tensor(h_m, dtype=torch.float32, device=device)
        
    if torch.is_tensor(dT):
        dT_clamped = torch.clamp(dT, -25.0, 15.0)
    else:
        dT_clamped = torch.clamp(torch.tensor(dT, device=device, dtype=torch.float32), -25.0, 15.0)
    
    # Correct tropospheric temperature calculation
    T0_shift = T0 + dT_clamped
    T = torch.clamp(T0_shift - 0.0065 * h_m, min=216.65 + dT_clamped)
    
    # Tropospheric density and pressure
    rhotrop = rho0 * torch.pow(T / T0_shift, 4.256848030018761)
    dhstrat = torch.clamp(h_m - 11000.0, min=0.0)
    rho = rhotrop * torch.exp(-dhstrat / 6341.552161)
    p = rho * R * T
    return p, rho, T

def tas2mach_torch(v_tas_mps, T_k):
    """Convert true airspeed (m/s) to Mach number in PyTorch."""
    gamma = 1.4
    R = 287.05287
    a = torch.sqrt(gamma * R * T_k)
    return v_tas_mps / a

def tas2cas_torch(v_tas_mps, p_pa, rho_kgm3):
    """Convert true airspeed (m/s) to calibrated airspeed (m/s) in PyTorch."""
    p0 = 101325.0
    rho0 = 1.225
    
    # Clamp input arguments to prevent overflow/underflow/nan
    p_clamped = torch.clamp(p_pa, min=10.0)
    rho_clamped = torch.clamp(rho_kgm3, min=1e-4)
    v_tas_mps_clamped = torch.clamp(v_tas_mps, min=0.0)
    
    qdyn_base = torch.clamp(1.0 + rho_clamped * v_tas_mps_clamped * v_tas_mps_clamped / (7.0 * p_clamped), min=1.0)
    qdyn = p_clamped * (torch.pow(qdyn_base, 3.5) - 1.0)
    v_cas_arg = 7.0 * p0 / rho0 * (torch.pow(torch.clamp(qdyn / p0 + 1.0, min=1e-6), 2.0 / 7.0) - 1.0)
    v_cas = torch.sqrt(torch.clamp(v_cas_arg, min=0.0))
    return v_cas


class PhysicsLoss(nn.Module):
    """
    Physics-Informed Loss for Latent Diffusion Models (PI-LDM).
    Bakes aerodynamics laws and OpenAP operational limits into the model training.
    """
    def __init__(self, 
                 ac_types=None,
                 dt: float = 1.0, 
                 gamma1: float = 1.0, 
                 gamma2: float = 1.0, 
                 gamma3: float = 1.0):
        super().__init__()
        self.dt = dt
        
        # Loss weights
        self.gamma1 = gamma1  # EOM weight
        self.gamma2 = gamma2  # Energy weight
        self.gamma3 = gamma3  # Envelope weight
        
        self.ac_types = ac_types
        self._cache_aircraft_properties()
        
    def _cache_aircraft_properties(self):
        """
        Dynamically queries OpenAP database at instantiation and caches properties 
        for each aircraft type to avoid slow disk read loops during training.
        """
        from openap import prop, Drag, Thrust, WRAP
        import warnings
        
        self.aircraft_cache = {}
        ac_list = self.ac_types if self.ac_types else ['a320']
        
        # Suppress OpenAP loading warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for ac in ac_list:
                ac_code = ac.lower()
                if ac_code not in prop.available_aircraft():
                    ac_code = 'a320'
                
                # Default A320 fallbacks
                S = 124.0
                cd0 = 0.018
                k = 0.039
                mass = 60000.0
                cruise_alt = 36089.23
                eng_bpr = 5.7
                eng_max_thrust = 117900.0
                eng_number = 2
                cruise_mach = 0.78
                eng_cruise_thrust = 27110.0
                v_stall = 60.0
                v_max = 180.0
                max_roc = 25.0
                
                try:
                    aircraft = prop.aircraft(ac_code)
                    limits = aircraft.get('limits', {})
                    mtow = limits.get('MTOW', 78000.0)
                    oew = limits.get('OEW', 42600.0)
                    mass = (mtow + oew) / 2.0
                    v_max = limits.get('VMO', 350.0) * 0.514444
                    
                    wing = aircraft.get('wing', {})
                    S = wing.get('area', 124.0)
                    
                    d = Drag(ac_code)
                    polar = d.polar
                    cd0 = polar.get('clean', {}).get('cd0', cd0)
                    k = polar.get('clean', {}).get('k', k)
                    
                    t = Thrust(ac_code)
                    cruise_alt = t.cruise_alt
                    eng_bpr = t.eng_bpr
                    eng_max_thrust = t.eng_max_thrust
                    eng_number = t.eng_number
                    cruise_mach = t.cruise_mach
                    eng_cruise_thrust = t.eng_cruise_thrust
                    
                    w = WRAP(ac_code)
                    v_stall = w.landing_speed().get('minimum', v_stall)
                except Exception:
                    pass
                
                self.aircraft_cache[ac] = {
                    'S': S,
                    'cd0': cd0,
                    'k': k,
                    'mass': mass,
                    'cruise_alt': cruise_alt,
                    'eng_bpr': eng_bpr,
                    'eng_max_thrust': eng_max_thrust,
                    'eng_number': eng_number,
                    'cruise_mach': cruise_mach,
                    'eng_cruise_thrust': eng_cruise_thrust,
                    'v_stall': v_stall,
                    'v_max': v_max,
                    'max_roc': max_roc
                }

    def forward(self, trajectories, cond=None):
        """
        Calculates the total physics loss over a set of generated trajectories.
        trajectories shape: (batch_size, seq_len, 4) -> [track, gs, alt, time]
        """
        # Ensure we work strictly with 4 features (track, gs, alt, time)
        trajectories = trajectories[..., :4]
        
        batch_size = trajectories.shape[0]
        device = trajectories.device
        
        total_eom = torch.tensor(0.0, device=device)
        total_energy = torch.tensor(0.0, device=device)
        total_envelope = torch.tensor(0.0, device=device)
        
        for i in range(batch_size):
            # 1. Resolve aircraft properties
            if cond is not None and self.ac_types is not None:
                ac_idx = int(torch.round(cond[i, 1]).item())
                ac_idx = max(0, min(ac_idx, len(self.ac_types) - 1))
                ac_name = self.ac_types[ac_idx]
            else:
                ac_name = list(self.aircraft_cache.keys())[0] if self.aircraft_cache else 'a320'
                
            props = self.aircraft_cache.get(ac_name, self.aircraft_cache.get('a320'))
            
            # Trajectory slice: (seq_len, 4)
            traj = trajectories[i]
            
            # Unpack features
            track = traj[:, 0]
            gs = traj[:, 1]
            alt = traj[:, 2]
            time = traj[:, 3]
            
            # Convert to physical SI units
            psi = track * (math.pi / 180.0)
            v_tas = gs * 0.514444
            h = alt * 0.3048
            t = time
            
            # Time differences
            dt = t[1:] - t[:-1]
            dt = torch.clamp(dt, min=0.1)
            
            # Compute physical derivatives
            dh = h[1:] - h[:-1]
            dv = v_tas[1:] - v_tas[:-1]
            dpsi = psi[1:] - psi[:-1]
            
            # Wrap heading rate angular differences cleanly
            dpsi = torch.atan2(torch.sin(dpsi), torch.cos(dpsi))
            
            roc = dh / dt       # Rate of climb/descent (m/s)
            acc = dv / dt       # Acceleration (m/s^2)
            omega = dpsi / dt   # Heading change rate (rad/s)
            
            # Midpoint states for forces and density calculations
            v_mid = torch.clamp(0.5 * (v_tas[1:] + v_tas[:-1]), min=5.0)
            h_mid = torch.clamp(0.5 * (h[1:] + h[:-1]), min=0.0)
            
            # 2. Dynamic Atmosphere
            p, rho, T = compute_atmos_torch(h_mid)
            
            # 3. Aerodynamic Drag Polar (Differentiable)
            S = props['S']
            cd0 = props['cd0']
            k = props['k']
            mass = props['mass']
            
            qS = 0.5 * rho * (v_mid ** 2) * S
            gamma = torch.atan2(roc, v_mid)
            L = mass * 9.80665 * torch.cos(gamma)
            cl = L / torch.clamp(qS, min=1e-3)
            cd = cd0 + k * (cl ** 2)
            Drag_force = cd * qS
            
            # 4. Engine Climb Thrust Limits (Differentiable Segment Model)
            tas_kt = v_mid / 0.514444
            alt_ft = h_mid / 0.3048
            roc_fpm = roc / 0.00508
            roc_abs = torch.abs(roc_fpm)
            
            Fcr = props['eng_cruise_thrust'] * props['eng_number']
            cruise_mach = props['cruise_mach']
            cruise_alt = props['cruise_alt']
            
            h_cr_m = torch.tensor(cruise_alt * 0.3048, device=device, dtype=torch.float32)
            p_cr, rho_cr, T_cr = compute_atmos_torch(h_cr_m)
            v_tas_ref_mps = cruise_mach * torch.sqrt(1.4 * 287.05287 * T_cr)
            vcas_ref = tas2cas_torch(v_tas_ref_mps, p_cr, rho_cr)
            
            mach = tas2mach_torch(v_mid, T)
            vcas = tas2cas_torch(v_mid, p, rho)
            
            mratio = mach / cruise_mach
            vratio = vcas / torch.clamp(vcas_ref, min=1.0)
            
            # Segment 3 (Alt > 30000 ft)
            d_coef = -0.4204 * mratio + 1.0824
            bcoef = torch.pow(torch.clamp(mratio, min=1e-3), -0.11)
            ratio_seg3 = d_coef * torch.log(torch.clamp(p / p_cr, min=1e-3)) + bcoef
            
            # Segment 2 (10000 ft < Alt <= 30000 ft)
            a_coef = torch.pow(torch.clamp(vratio, min=1e-3), -0.1)
            n_coef = 2.667e-05 * roc_abs + 0.8633
            ratio_seg2 = a_coef * torch.pow(torch.clamp(p / p_cr, min=1e-3), -0.355 * vratio + n_coef)
            
            # Segment 1 (Alt <= 10000 ft)
            p_10 = torch.tensor(3048.0, device=device, dtype=torch.float32)
            p_10_val, _, _ = compute_atmos_torch(p_10)
            F10 = Fcr * a_coef * torch.pow(torch.clamp(p_10_val / p_cr, min=1e-3), -0.355 * vratio + n_coef)
            m_coef = -1.2043e-1 * vratio - 8.8889e-9 * (roc_abs ** 2) + 2.4444e-5 * roc_abs + 4.7379e-1
            ratio_seg1 = m_coef * (p / p_cr) + (F10 / Fcr - m_coef * (p_10_val / p_cr))
            
            climb_ratio = torch.where(
                alt_ft > 30000.0, ratio_seg3,
                torch.where(alt_ft > 10000.0, ratio_seg2, ratio_seg1)
            )
            Thrust_max = climb_ratio * Fcr
            
            # 5. Engine Descent Idle Thrust Limits (Takeoff-based approximation)
            G0 = 0.0606 * props['eng_bpr'] + 0.6337
            dP_atm = p / 101325.0
            A_coef = -0.4327 * (dP_atm ** 2) + 1.3855 * dP_atm + 0.0472
            Z_coef = 0.9106 * (dP_atm ** 3) - 1.7736 * (dP_atm ** 2) + 1.8697 * dP_atm
            X_coef = 0.1377 * (dP_atm ** 3) - 0.4374 * (dP_atm ** 2) + 1.3003 * dP_atm
            takeoff_ratio = A_coef - 0.377 * (1.0 + props['eng_bpr']) / math.sqrt((1.0 + 0.82 * props['eng_bpr']) * G0) * Z_coef * mach + (0.23 + 0.19 * math.sqrt(props['eng_bpr'])) * X_coef * (mach ** 2)
            Thrust_takeoff = takeoff_ratio * props['eng_max_thrust'] * props['eng_number']
            Thrust_idle = 0.07 * Thrust_takeoff
            
            # --- Loss Computations ---
            
            # A. Equation of Motion (Kinematic bounds check)
            acc_max = (Thrust_max - Drag_force) / mass
            acc_min = (Thrust_idle - Drag_force) / mass
            loss_eom_i = torch.mean(torch.square(torch.relu(acc - acc_max)) + torch.square(torch.relu(acc_min - acc)))
            total_eom = total_eom + loss_eom_i
            
            # B. Energy Conservation (Power Balance Check)
            T_req = Drag_force + mass * (9.80665 * roc / v_mid + acc)
            loss_energy_i = torch.mean(torch.square(torch.relu(T_req - Thrust_max)) + torch.square(torch.relu(Thrust_idle - T_req)))
            total_energy = total_energy + loss_energy_i
            
            # C. Flight Envelope (Stall, Max Speed, Vertical Rate, Bank Angle)
            v_stall = props['v_stall']
            v_max = props['v_max']
            max_roc_val = props['max_roc']
            
            loss_stall = torch.square(torch.relu(v_stall - v_mid))
            loss_vmax = torch.square(torch.relu(v_mid - v_max))
            loss_roc = torch.square(torch.relu(torch.abs(roc) - max_roc_val))
            
            # Bank angle: phi = atan(V * turn_rate / g0)
            bank_angle = torch.atan2(v_mid * omega, torch.tensor(9.80665, device=device))
            loss_bank = torch.square(torch.relu(torch.abs(bank_angle) - 0.5236)) # Max bank 30 degrees (0.5236 rad)
            
            loss_envelope_i = torch.mean(loss_stall + loss_vmax + loss_roc + loss_bank)
            total_envelope = total_envelope + loss_envelope_i
            
        # Average over batch size
        total_eom = total_eom / batch_size
        total_energy = total_energy / batch_size
        total_envelope = total_envelope / batch_size
        
        l_physics = (self.gamma1 * total_eom) + (self.gamma2 * total_energy) + (self.gamma3 * total_envelope)
        return l_physics
