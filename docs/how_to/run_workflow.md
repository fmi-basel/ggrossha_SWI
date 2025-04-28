# Single Worm Imaging Workflow

When you run this workflow, all your SWI experiments and results will be organized in the `ggrossha_SWI-Template` folder on Tachyon:

- `raw_data`: symbolic links to the original raw image folders (before writing to tape, please write to tape as soon as you can, and delete the raw data folder!)
- `/runs`: contains the selection.csv file
- `/processed_data/s01_zarr_data`: compressed Zarr images from raw data (can this stay?)
- `/processed_data/s02_segment`: segmentation masks and quality control images from the neural network
- `/processed_data/s03_measurements`: extracted worm features (size, intensity over time) plus quality control plots
- `/results`: final summary plots and analyses


---

## Overview

The analysis consists of:

1. **Collect/locate Data** [Server]: create a symbolic link to your raw image folders
2. **Create Preview** [Server]: generate downsampled previews of worms
3. **Create Selection** [Local]: annotate hatch, escape, and validity in Napari
4. **Segmentation & Quantification** [Server]: perform full Zarr conversion, segmentation, and feature extraction
5. **Inspect different image channels** [Local]: review fluorescent images and optionally annotate molt events
6. **Plot Results** [Local]: plot quantified GFP intensity

!!! note "SLURM vs Local Processing"
    Operations tagged **[Server]** run on the SLURM cluster via SSH, while those tagged **[Local]** run on your own computer.


??? info "Server Setup & Monitoring"  
    1. Open your terminal or command prompt.  
    2. SSH into the SLURM head node:  
       ```bash
       ssh <your-username>@vcl1062.fmi.ch
       ```  
    3. Change directory to the analysis repository:  
       ```bash
       cd /tachyon/scratch/ggrossha/<your-username>/.../ggrossha_SWI-Template
       ```  
    4. Initialize the shell:
        ```commandline
        source init.sh
        ```
    5. To check running or queued jobs:  
       ```bash
       squeue -u <your-username>
       ```
       

??? info "Local Setup"  
    1. Open your terminal or command prompt.  
    2. Change directory to your local clone of the repository:  
       ```bash
       cd /path/to/ggrossha_SWI-Template
       ```

---

??? Flowchart
    The following flowchart shows the steps involved in the process:
    ```mermaid
    flowchart TD
      %% Main steps
      A[Collect/locate Data] --> B[Create Preview]
      B --> C{Annotate hatch and escape}
      C --> D[Run workflow]
    

      %% Sub-graph for “Run workflow”
      subgraph "Inside the workflow"
        D1[Compress data]
        D2[Segment worms]
        D3[Quantify features]
      end

      %% Connect sub-steps into the main flow
      D --> D1
      D1 --> D2
      D2 --> D3
      D3 -->E{Plot Results}
      D1 -->F{inspect fluorescent images}

    ```

---


## 1. Collect/Locate data **[SERVER]**
We want to keep track of which raw data was used as part of this project. To facilitate this, we will create a symbolic link to the raw data in the `raw_data` directory.


Link the acquisition data:
```commandline
ACQUIRED_DATA=/path/to/raw_data pixi run link_dataset
```

??? info "Step-by-Step Instructions"
    1. **Open your terminal or command prompt.**
    2. **SSH into the server:  **
       ```bash
       ssh <your-username>@vcl1062.fmi.ch
       ```  
    3. **Change directory to the analysis repository**:
       ```bash
       cd /tachyon/scratch/ggrossha/<your-username>/.../ggrossha_SWI-Template
       ```  
    4. **Initialize the shell**:
        ```commandline
        source init.sh
        ```
    5. **Run the following command**:
       ```commandline
       ACQUIRED_DATA=/path/to/raw_data pixi run link_dataset
       ```

---

## 2. Create Preview **[SERVER]**
The preview is a reduced representation of the raw data. It is used to pre-select the data that will be used for the analysis.

Please fill out the following form with the appropriate values (choose for slurm account`gross-hcs` or `grossfold`):

{{{user-defined-values}}}

```commandline
WD=runs/DATASET ACCOUNT=SLURM_ACCOUNT pixi run convert_to_zarr_slurm
```

??? info "Step-by-Step Instructions"
    After running the command
    ```commandline
    WD=runs/DATASET ACCOUNT=SLURM_ACCOUNT pixi run convert_to_zarr_slurm
    ```
    the terminal will prompt you with several questions:
    Follow these steps carefully:

    1. **Select the task (`s01_convert_to_zarr`)**  
        - Use the **arrow keys** to highlight `s01_convert_to_zarr`.
        - Press **space** to select it.
        - Press **enter** to proceed.

    2. **Provide the path to the raw data**  
        - Navigate to the (linked) raw data folder.
        - Press **Tab** to autocomplete or select the correct path.

    3. **Answer the prompts**:

        | Prompt | Answer | Notes |
        |:-------|:-------|:------|
        | List channels to process | `1` | Channel 1 is brightfield. For annotating hatch and escape, only brightfield is needed. |
        | YX bin factor | `2` | Reduces file size; full resolution is not necessary for preview. |
        | Create MIPs? | `Yes` | Recommended; MIP view is sufficient for annotation. |
        | Convert legacy compressed TIFF format? | `Yes` | Ensures file compatibility. |
        | YX spacing (microns) | _(Check with Marien or Johnson)_ | Based on microscope settings. |
        | Z spacing (microns) | _(Use microscope setting)_ | Based on imaging setup. |

    > **Note:**  
     Always double-check the YX and Z spacing values to match the microscope settings used during acquisition.

---


## 3. Create Selection **[Local]**

This step is about manually reviewing your microscopy experiment to annotate **hatch**, **escape**, and **validity** for each worm chamber. You’ll do this using a Jupyter Notebook and the napari viewer. The results will be saved to a folder of your choice (e.g., `runs/20250101_GeneX`), and the process takes around **30 minutes-1 hour**.

> **Note:**  
 Pay attention, this is local!!!

=== "Linux and macOS"

    ```bash
    WD=runs/DATASET pixi run create_selection
    ```

=== "Windows (PowerShell)"

    ```powershell
    $env:WD='runs/DATASET'; pixi run create_selection
    ```

??? info "Step-by-Step Instructions"
    1. **Open the Terminal**

        Navigate to your project folder:

        === "Linux and macOS"

            ```bash
            cd <path_in_your_computer_to>/ggrossha_SWI-Template
            ```

        === "Windows (PowerShell)"

            ```powershell
            cd <path_in_your_computer_to>\ggrossha_SWI-Template
            ```

        Replace `<path_in_your_computer_to>` with the full path where you cloned the repository.


    2. **Create Output Folder and Launch the Analyzer**

        Choose a name for your analysis folder, like `20250101_GeneX`.

        Run:

        === "Linux and macOS"

            ```bash
            WD=runs/DATASET pixi run create_selection
            ```

        === "Windows (PowerShell)"

            ```powershell
            $env:WD='runs/DATASET'; pixi run create_selection
            ```

        Example:  
        `WD=runs/20250101_GeneX pixi run create_selection`

        This will:
        - Create the folder
        - Copy the notebook
        - Launch Jupyter Lab


    3. **Run the First Two Cells in the Notebook**

        These cells will:
        - Import packages
        - Open napari viewer



    4. **Select the Processing Directory**

        In the **Jupyter Notebook**:

        ```bash
        /tachyon/scratch/gmicro_prefect/ggrossha/ggrossha_SWI-Template
        ```

        Press "Change", then run the cell.



    5. **Select the Experiment**

        - Run the first cell
        - Choose the experiment
        - Run the next three cells



    6. **Annotate All Positions in napari**

        Use keyboard shortcuts:

        | Key | Action           | Description                         |
        |-----|-------------------|-------------------------------------|
        | `S` | Skip               | Marks worm as invalid               |
        | `E` | Set start          | Marks hatch                        |
        | `R` | Set end            | Marks escape or contamination      |
        | `W` | Next worm          | Move to next worm                  |
        | `Q` | Previous worm      | Move to previous worm              |


    7. **Save the Annotation File**

        Run the last two cells to:
        - Save the annotations
        - Export `.csv` file



    8. **Add Conditions and Notes**

        Edit the `.csv` file:
        - Add condition names
        - Add any relevant notes

---
## 4. Compression, Segmentation and Quantification **[SERVER]**

Once a `selection.csv` has been made, the main workflow can be started.  
The workflow will always **compress** the raw data into OME-Zarr format (TIFF compressed).  
**Optionally**, it can also **perform segmentation and quantification** if a segmentation config is provided.


 Choose your workflow:

| Goal | Instructions |
|:-----|:-------------|
| **Only compress** raw data | ➔ Skip config, directly run the workflow |
| **Compress + segment + quantify** | ➔ First create a segmentation config, then run the workflow |


??? info "Step-by-Step Instructions only compressions"
    If you only want compression (no segmentation, no quantification):

    ```commandline
    WD=runs/DATASET ACCOUNT=SLURM_ACCOUNT MAIL=MAIL_TO pixi run submit_workflow
    ```

??? info "Step-by-Step Instructions compress + segment + quantify"
    ##### Create the config file

    Run:

    ```commandline
    WD=runs/DATASET pixi run config
    ```

    Follow the prompts:

    1. **Select the task (`s02_segment`)**  
        - Use the **arrow keys** to highlight `s02_segment`.
        - Press **space** to select it.
        - Press **enter** to proceed.

    2. **Provide the path to the data directory**  
        - Path: `/processed_data/DATASET/s01_zarr_data`  
        - (Auto-filled in most cases.)

    3. **Provide the path to the output directory**  
        - Path: `/processed_data/DATASET/`  
        - (Auto-filled in most cases.)

    4. **Provide the path to the segmentation model checkpoint**  
        - Example:  
          `/tachyon/groups/scratch/gmicro_prefect/ggrossha/ancneagu/ggrossha_GRH1/processed_data/20250328_grh-1_reporter_new_RNAi/segmentation_model/epoch=54-step=9515.ckpt`

    5. **Brightfield channel index**  
        - Use `0` (brightfield is usually channel 0).

    6. **Batch size**  
        - Standard is `10`.  
        - (Increase if you want more parallelization.)

    ---
    #### Start the workflow

    After the config is created, submit the workflow:

    ```commandline
    WD=runs/DATASET ACCOUNT=SLURM_ACCOUNT MAIL=MAIL_TO pixi run submit_workflow
    ```


---

## 5. Inspect different channels and/or annotate molt [Local]
(how?!)

A jupyter notebook is provided to annotate molt points.

=== "Linux and macOS"

    ```bash
    WD=runs/DATASET pixi run annotate_molt
    ```

=== "Windows (PowerShell)"

    ```powershell
    $env:WD='runs/DATASET'; pixi run annotate_molt
    ```


## 6. Plot Results [Local]
A jupyter notebook is provided to visualize the results.

=== "Linux and macOS"

    ```bash
    WD=runs/DATASET pixi run plot
    ```

=== "Windows (PowerShell)"

    ```powershell
    $env:WD='runs/DATASET'; pixi run plot
    ```



