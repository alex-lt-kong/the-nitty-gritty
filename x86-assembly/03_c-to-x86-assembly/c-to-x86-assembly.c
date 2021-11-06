int main() {

  int a = 27;
  int b = a + 5;

  return 0;
}

// gcc -O0 -S c-to-x86-assembly.c -masm=intel 