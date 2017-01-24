
WestGrid is a type of cloud service, belongs to Compute Canada

* Login
 * ssh orcinus.westgrid.ca
 * ssh [your user name]@orcinus.westgrid.ca
 
* Use python3.5 when the default version is not
 * Type `module load python/3.5.0` in their Linux terminal
 * Type `python3.5`

* Sending files/folder between my machine and WestGrid machine, have to use `:` colon at the end, otherwise I'm making local copy
 * For example, `scp -r [My file path on my machine]/SFU_comments_extractor [user name]@orcinus.westgrid.ca:`
