# Analysis Environment Setup

This document describes how to install and configure the software environment required for the single-worm imaging analysis.

## 1. Prerequisites

Before you begin, ensure that you have the following:

- **Git (version ≥2.20)**  
  Follow the instructions at [Install Git](https://github.com/git-guides/install-git) to install Git on your platform.
- **Command-line shell**  
    - **macOS:** Terminal (bash or zsh)  
    - **Windows:** PowerShell or Command Prompt (cmd)
- **Network access**  
  Internet and FMI internal network access (VPN or on-campus) to clone repositories and download dependencies.
- **GitHub credentials**  
  Username + personal access token.
- **Server access**  
  - **SSH to `vcl1062`:**  
    ```bash
    ssh <your-username>@vcl1062.fmi.ch
    ```
  - **SLURM cluster account:**  
    - `gross-hcs` or `grossfold`  
    - To request an account, email [help.IT@fmi.ch](mailto:help.IT@fmi.ch)  
    - More info: [FMI Compute Cluster Wiki](https://wiki.fmi.ch/display/FMICOMP/Compute+Cluster)

## 2. Clone the Analysis Repository. Clone the Analysis Repository

Clone the repository on **both**:

- **your local machine**
- **remote server** (`/tachyon/scratch/grossha/<your-username>`)


To do this, run in your terminal/command prompt:

```bash
cd /path/to/desired/location
git clone https://github.com/fmi-basel/ggrossha_SWI-Template.git
```

**Troubleshooting:**

Authentication fails: re-run with your personal access token:

```bash
git clone https://<your-username>:<your-token>@github.com/fmi-basel/ggrossha_SWI-Template.git
```

Need or lost a token? Visit *Settings → Developer settings → Personal access tokens* on GitHub.

## 3. Install Pixi

Pixi is the package manager for the SWI-Template. Install it in the project directory on BOTH locally and on the server:

- **macOS/Linux:**
  ```bash
  cd ggrossha_SWI
  source install.sh
  ```
- **Windows (PowerShell):**
  ```powershell
  cd ggrossha_SWI-Template
  powershell -ExecutionPolicy ByPass -c "irm -useb https://pixi.sh/install.ps1 | iex"
  ```

## 4. Configure Cache and Initialize

1. **Create the cache directory** (required for micro_sam data):
   ```bash
   mkdir -p ~/.cache/micro_sam/
   ```
2. **Source the initialization script** to set up environment variables and generate the site:
   ```bash
   source init.sh
   ```

After running `source init.sh`, a `site/` folder will appear in your analysis folder. Navigate into this folder and open `index.html` in your browser to load the analysis home page.

## 5. Launch and Update the Site

- **To view the site:** open `site/index.html` in your browser.
- **To update to the latest version:**
  ```bash
  git pull origin main
  source init.sh
  ```

---

*Next:* proceed to **Data Management & Basic Analysis** steps in the tutorial.

