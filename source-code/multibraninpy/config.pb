language: PYTHON
name:     "branin"

# Task indicator
variable {
 name: "Task"
 type: INT
 size: 1
 min:  0
 max:  2
}

variable {
 name: "X"
 type: FLOAT
 size: 2
 min:  0
 max:  1
}

# Integer example
#
# variable {
#  name: "Y"
#  type: INT
#  size: 5
#  min:  -5
#  max:  5
# }

# Enumeration example 
# variable {
#  name: "Z"
#  type: ENUM
#  size: 3
#  options: "foo"
#  options: "bar"
#  options: "baz"
#}

# Enumeration example 
#variable {
#  name: "Categories"
#  type: ENUM
#  size: 2
#  options: "foo"
#  options: "bar"
#}
