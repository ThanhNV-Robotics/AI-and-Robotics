from cartPoledynamics import *
import yaml
from DDP import DDP

if __name__ == "__main__":

    # Load parameters
    with open('DDP/cartPole/config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    T  = config['DDP']['T']
    h  = config['dynamic_parameters']['h']
    Nt = int(T / h)

    print(f"T={T}s  h={h}s  Nt={Nt}")

    # Cart-pole model
    model = cartPole(config['dynamic_parameters'])

    # Initial and target states: [x, theta, dx, dtheta]
    x0      = np.array([0.0, 0.0, 0.0, 0.0])
    xtarget = np.array([0.0, np.pi, 0.0, 0.0])
    dim_x   = x0.shape[0]

    # DDP solver
    ddp_solver = DDP(
        config   = config['DDP'],
        Nx       = dim_x,
        Nu       = 1,
        Nt       = Nt,
        xgoal    = xtarget,
        dynamics = model.DiscreteEulerDynamics,
    )
    print("DDP solver ready.")

    # Warm-start initial guess from inverse dynamics
    X_guess, _, U_guess, _ = model.GenWarmSwingUp(np.zeros([dim_x, Nt]), T, Nt)
    
    # GenWarmSwingUp returns X_guess (Nt+1, 4) and U_guess (Nt,)
    # DDP convention: X is (Nx, Nt), U is (Nu, Nt)
    X_init = 0*X_guess[:Nt, :].T            # (4, Nt)
    U_init = 0*U_guess[:Nt].reshape(1, -1)  # (1, Nt)

    # Run DDP optimisation
    X_opt, U_opt, J_final, cost_history, X_history, U_history = ddp_solver.optimize(x0, X_init, U_init)

    # Plot results
    t = np.linspace(0, T, Nt)

    # model.simulate_state_trajectory(t, X_guess, isRender=True)

    labels      = ['x (m)', 'θ (rad)', 'ẋ (m/s)', 'θ̇ (rad/s)']
    target_vals = xtarget

    fig, axes = plt.subplots(3, 2, figsize=(12, 9))
    fig.suptitle(f"DDP Swing-up  (final J={J_final:.3f})")

    for i, (ax, label) in enumerate(zip(axes.flat[:4], labels)):
        ax.plot(t, X_opt[i, :],  label='DDP opt')
        ax.plot(t, X_init[i, :], '--', alpha=0.5, label='Init guess')
        ax.axhline(target_vals[i], color='r', linestyle=':', linewidth=1, label='Target')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel(label)
        ax.legend(fontsize=7)
        ax.grid(True)

    axes[2, 0].plot(t, U_opt[0, :],  label='DDP opt')
    axes[2, 0].plot(t, U_init[0, :], '--', alpha=0.5, label='Init guess')
    axes[2, 0].set_xlabel('Time (s)')
    axes[2, 0].set_ylabel('u (N)')
    axes[2, 0].set_title('Control input')
    axes[2, 0].legend(fontsize=7)
    axes[2, 0].grid(True)

    axes[2, 1].plot(cost_history, marker='o', markersize=3)
    axes[2, 1].set_xlabel('Iteration')
    axes[2, 1].set_ylabel('Cost J')
    axes[2, 1].set_title('Cost vs. Iteration')
    axes[2, 1].set_yscale('log')
    axes[2, 1].grid(True)
    plt.tight_layout()
    plt.show()

    # --- Trajectory evolution across iterations ---
    n_iters = len(X_history)
    cmap    = plt.cm.plasma
    colors  = [cmap(v) for v in np.linspace(0.1, 0.95, n_iters)]

    fig2, axes2 = plt.subplots(3, 2, figsize=(12, 9))
    fig2.suptitle("Trajectory evolution across DDP iterations")

    for idx, (Xh, Uh) in enumerate(zip(X_history, U_history)):
        c     = colors[idx]
        alpha = 0.3 + 0.7 * (idx / max(n_iters - 1, 1))   # early=faint, final=opaque
        lw    = 0.8 + 1.2 * (idx / max(n_iters - 1, 1))
        label = f"iter {idx}" if idx in (0, n_iters - 1) else None
        for i, ax in enumerate(axes2.flat[:4]):
            ax.plot(t, Xh[i, :], color=c, alpha=alpha, linewidth=lw, label=label)
        axes2[2, 0].plot(t, Uh[0, :], color=c, alpha=alpha, linewidth=lw, label=label)

    for i, (ax, label) in enumerate(zip(axes2.flat[:4], labels)):
        ax.axhline(target_vals[i], color='r', linestyle=':', linewidth=1, label='Target')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel(label)
        ax.legend(fontsize=7)
        ax.grid(True)

    axes2[2, 0].set_xlabel('Time (s)')
    axes2[2, 0].set_ylabel('u (N)')
    axes2[2, 0].set_title('Control input')
    axes2[2, 0].legend(fontsize=7)
    axes2[2, 0].grid(True)

    # Colorbar to indicate iteration progress
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, n_iters - 1))
    sm.set_array([])
    fig2.colorbar(sm, ax=axes2[2, 1], label='Iteration')
    axes2[2, 1].axis('off')

    plt.tight_layout()
    plt.show()

    # Animate the optimised trajectory
    model.simulate_state_trajectory(t, X_opt.T, isRender=True)
