
These files detail how we create the concept and visual programs for our layout domain.

**man_concepts.py** --> contains logic describing the *manufacturing* meta-concepts

**org_concepts.py** --> contains logic describing the *organic* meta-concepts 

**make_data.py** --> contains logic for how we sample the above logic to make a target dataset

**lay_data.json** --> is the output of make_data.py that we use for our experiments

**data_split.py** --> contains the split of the above concepts into a train / val / test set according to their attributes

To recreate new data from these concepts, use a command like:

```
python3 make_data.py {OUT_DIR} {NUM_PER_CONCEPT}
```