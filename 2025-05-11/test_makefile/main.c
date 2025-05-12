#include <stdio.h>
#include "factorial.h"
#include "printhello.h"

int main() {
    printhello();
    printf("This is a main\n");
    printf("The factorial of 6 is: %d\n", factorial(6));
    return 0;
}
