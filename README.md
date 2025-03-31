# Single Worm Imaging - Analysis Template
This template repository contains the first two processing steps for any Single Worm Imaging project. In the first step, the raw data is converted to ome-zarr and in the second step the worm segmentation is performed. Further processing steps can be added to the workflow.

__HELP__: Please reach out to FAIM if you have any questions or need help with setting up the project.

## Create a Copy of this Repository
This repository is a template repository. To create a copy of this repository, click on the green "Use this template" button on the top right of the repository page.

## Clone the Repository
After creating a copy of the repository, clone the repository to your local machine and to the fileserver. The copy on the fileserver is used for the processing of the data, whereas the local copy is used to visualize and annotate the data.

## Documentation
Once the repository is cloned, the documentation can be rendered with the following command:

```commandline
pixi run build_docs
```

Running this command for the first time can take a while. After the command has finished, the documentation can be viewed by opening the `sites/index.html` file in a web browser. To run the processing follow the `How-to Guide` section in the documentation.

---
This project was generated with the [faim-ipa-project](https://fmi-faim.github.io/ipa-project-template/) copier template.
