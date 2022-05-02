=======================================================
Convert `highD dataset <https://www.highd-dataset.com/>`_ to `CommonRoad .xml format <https://gitlab.lrz.de/tum-cps/commonroad-scenarios/-/blob/master/documentation/XML_commonRoad_2018b.pdf>`_
=======================================================

System Requirements
-------------------

The software is written in Python 3.6 and tested on Linux. The usage of the Anaconda_ Python distribution is strongly recommended.

.. _Anaconda: http://www.anaconda.com/download/#download


Dependencies
------------

The required Python dependencies are:

* numpy>=1.13
* pandas
* commonroad-io==2020.2


#. Activate your environment with 

	.. code-block:: console

		   $ source activate cr36
		   

#. Change the source code of ``make_valid_orientation`` function in ``commonroad/common/util.py`` to:

    .. code-block:: python

        def make_valid_orientation(angle: float) -> float:
            while angle > 2 * np.pi - 5e-5:
                angle = angle - 2 * np.pi
            while angle < -2 * np.pi + 5e-5:
                angle = angle + 2 * np.pi
            return angle



   This is due to a known bug of commonroad-io: After rotating the scenario, the orientation of some vehicles exceed 2pi. The reason is that the ``make_valid_orientation`` function in commonroad-io does not convert an orientation which is slightly smaller than 2pi. But ``CommonRoadFileWriter`` round this orientation to 4 digits which is larger than 2*pi.


#. Convert the dataset with

    .. code-block:: console
    
		   $ python highd_to_cr.py -i /path_to_your_highD_dataset/highD-dataset-v1.0/ -o /path_to_your_output_folder/cr_scenarios -np num_of_planning_problems_per_scenario

#. (Optional) If you want to accelerate the converting process with multiple cpu threads, change ``NUM_CPUS`` and ``HIGHD_DIR`` in ``mpi_processing.sh`` and run

    .. code-block:: console
    
           $ ./mpi_processing.sh

  
   Note that ``NUM_CPUS`` needs to be a divisor of 60 since the script divides the 60 recording files in ``NUM_CPUS`` processes.
 
