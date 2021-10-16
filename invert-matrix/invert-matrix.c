#include<stdio.h>
#include <curses.h> 
#include<math.h>
#include <sys/time.h>

#define SIZE 401

int main() {
	float a[SIZE][SIZE], x[SIZE], ratio;
	int i, j, k, n;
	FILE *fp = fopen("invert-matrix.in","r");

	printf("Enter order of matrix: ");
	scanf("%d", &n);

	printf("Enter coefficients of Matrix:\n");
	for(i=1;i<=n;i++) {
	  for(j=1;j<=n;j++) {
	   fscanf(fp, "%f", &a[i][j]);
		 //printf("%f ", a[i][j]);
	  }
		//	printf("\n");
	}
	fclose(fp);

	struct timeval stop, start;
	gettimeofday(&start, NULL);
	/* Augmenting Identity Matrix of Order n */
	for(i=1;i<=n;i++) {
	  for(j=1;j<=n;j++) {
		  a[i][j+n] = i==j ? 1 : 0;				   
		  }
	}
	
	/* Applying Gauss Jordan Elimination */
	int dn = 2 * n;
	for(i=1;i<=n;i++) {
		float aii_div = 1.0 / a[i][i] ;
		for(j=1; j<=n; j++) {
			if(i==j)
				continue;
			ratio = a[j][i] * aii_div ;
			for(k=1;k<=dn;k++) {
				a[j][k] -= ratio*a[i][k];
			}					
		}
	}
	
	/* Row Operation to Make Principal Diagonal to 1 */
	float ai_div[n];
	for(i=1;i<=n;i++) {
		ai_div[i] = 1.0 / a[i][i];
	}
	for(i=1;i<=n;i++) {			  
	  for(j=n+1;j<=dn;j++) {	
	   	a[i][j] *= ai_div[i];
	  }
	}
	gettimeofday(&stop, NULL);
	
	/* Displaying Inverse Matrix */
	printf("\nInverse Matrix is:\n");
	for(i=1;i<=n;i++) {
		if (i == 5) printf(".");
		else if (i > 5) continue;	 
		for(j=n+1;j<=dn;j++) {
			if (j < n+10) printf("%f ",a[i][j]);
			else if (j < n + 15) printf(".");
		}
		printf("\n");
	}
	printf("%f ms\n", ((stop.tv_sec - start.tv_sec) * 1000000 + stop.tv_usec - start.tv_usec) / 1000.0);
	return(0);
}