#include<stdio.h>
void hello(){
	printf("hello,world!!\n");
}
int main(){
	int arr[4]={1,2,3,4};
	for(int i=0;i<4;i++){
		printf("%d\n",arr[i]);
	}
	hello();
	return 0;
}
