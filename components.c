// Nilanjana Das
//ndas2@umbc.edu

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "dc_image.h"



#define MIN(a,b)  ( (a) < (b) ? (a) : (b) )
#define MAX(a,b)  ( (a) > (b) ? (a) : (b) )
#define ABS(x)    ( (x) <= 0 ? 0-(x) : (x) )


//--------------------------------------------------
//--------------------------------------------------
// You must modify this disjoint set implementation
//--------------------------------------------------
//--------------------------------------------------

struct DisjointSet;

struct Node {
    int gaussian_x,gaussian_y;
    struct Node* next;
};

struct Node1 {
    int label;
	int x_coordinate;
	int y_coordinate;
    struct Node1* next1;
};

typedef struct DisjointSet {
	int r,g,b;
	int x,y;
	int rank;
	int label;
	int gaussian_r,gaussian_g,gaussian_b;
	int gaussian_x,gaussian_y;
	struct DisjointSet *parent;
} DisjointSet;


// Path Compression
DisjointSet *DisjointSetFindRoot(DisjointSet *curr)
{

   if ((curr->parent->x == curr->x) && (curr->parent->y == curr->y))
    {
        return curr;
    }
    else
    { 
        DisjointSet *c = DisjointSetFindRoot(curr->parent);
        curr->parent = c;
        return c;
     }
	return NULL;
}

//Union by Rank
void DisjointSetUnion(DisjointSet *a, DisjointSet *b)
{
    DisjointSet* aRoot = DisjointSetFindRoot(a);
    DisjointSet* bRoot = DisjointSetFindRoot(b);
 
    if ((aRoot->x == bRoot->x) && (aRoot->y == bRoot->y))
        return;

    int aRank = aRoot->rank;
    int bRank = bRoot->rank;

    if (aRank < bRank) 
    {
        aRoot->parent = bRoot;
    }
    else if (bRank < aRank) 
    {
        bRoot->parent = aRoot;
    } 
    else
    {
        aRoot->parent = bRoot;
        bRoot->rank = bRoot->rank + 1;
    }
}

//--------------------------------------------------
//--------------------------------------------------
// The following "run" function runs the entire algorithm
//  for a single vision file
//--------------------------------------------------
//--------------------------------------------------

void run(const char *infile, const char *outpre, int canny_thresh, int canny_blur, int gridSize, int iStride, int jStride, int sigmaVal)
{
	int y,x,i,j;
	int rows, cols, chan;

	//-----------------
	// Read the image    [y][x][c]   y number rows   x cols  c 3
	//-----------------
	byte ***img = LoadRgb(infile, &rows, &cols, &chan);
	printf("img %p rows %d cols %d chan %d\n", img, rows, cols, chan);

	char str[4096];
	sprintf(str, "out/%s_1_img.png", outpre);
	SaveRgbPng(img, str, rows, cols);
	
	//-----------------
	// Convert to Grayscale
	//-----------------
	byte **gray = malloc2d(rows, cols);
	for (y=0; y<rows; y++){
		for (x=0; x<cols; x++) {
			int r = img[y][x][0];   // red
			int g = img[y][x][1];   // green
			int b = img[y][x][2];   // blue
			gray[y][x] =  (r+g+b) / 3;
		}
	}

	sprintf(str, "out/%s_2_gray.png", outpre);
	SaveGrayPng(gray, str, rows, cols);

	//-----------------
	// Box Blur   ToDo: Gaussian Blur is better
	//-----------------
	
	// Box blur is separable, so separately blur x and y
	int k_x=canny_blur, k_y=canny_blur;
	
	// blur in the x dimension
	byte **blurx = (byte**)malloc2d(rows, cols);
	for (y=0; y<rows; y++) {
		for (x=0; x<cols; x++) {
			
			// Start and end to blur
			int minx = x-k_x/2;      // k_x/2 left of pixel
			int maxx = minx + k_x;   // k_x/2 right of pixel
			minx = MAX(minx, 0);     // keep in bounds
			maxx = MIN(maxx, cols);
			
			// average blur it
			int x2;
			int total = 0;
			int count = 0;
			for (x2=minx; x2<maxx; x2++) {
				total += gray[y][x2];    // use "gray" as input
				count++;
			}
			blurx[y][x] = total / count; // blurx is output
		}
	}
	
	sprintf(str, "out/%s_3_blur_just_x.png", outpre);
	SaveGrayPng(blurx, str, rows, cols);
	
	// blur in the y dimension
	byte **blur = (byte**)malloc2d(rows, cols);
	for (y=0; y<rows; y++) {
		for (x=0; x<cols; x++) {
			
			// Start and end to blur
			int miny = y-k_y/2;      // k_x/2 left of pixel
			int maxy = miny + k_y;   // k_x/2 right of pixel
			miny = MAX(miny, 0);     // keep in bounds
			maxy = MIN(maxy, rows);
			
			// average blur it
			int y2;
			int total = 0;
			int count = 0;
			for (y2=miny; y2<maxy; y2++) {
				total += blurx[y2][x];    // use blurx as input
				count++;
			}
			blur[y][x] = total / count;   // blur is output
		}
	}
	
	sprintf(str, "out/%s_3_blur.png", outpre);
	SaveGrayPng(blur, str, rows, cols);
	
	
	//-----------------
	// Take the "Sobel" (magnitude of derivative)
	//  (Actually we'll make up something similar)
	//-----------------
	
	byte **sobel = (byte**)malloc2d(rows, cols);
	
	for (y=0; y<rows; y++) {
		for (x=0; x<cols; x++) {
			int mag=0;
			
			if (y>0)      mag += ABS((int)blur[y-1][x] - (int)blur[y][x]);
			if (x>0)      mag += ABS((int)blur[y][x-1] - (int)blur[y][x]);
			if (y<rows-1) mag += ABS((int)blur[y+1][x] - (int)blur[y][x]);
			if (x<cols-1) mag += ABS((int)blur[y][x+1] - (int)blur[y][x]);
			
			int out = 3*mag;
			sobel[y][x] = MIN(out,255);
		}
	}
	
	
	sprintf(str, "out/%s_4_sobel.png", outpre);
	SaveGrayPng(sobel, str, rows, cols);
	
	//-----------------
	// Non-max suppression
	//-----------------
	byte **nonmax = malloc2d(rows, cols);    // note: *this* initializes to zero!
	
	for (y=1; y<rows-1; y++)
	{
		for (x=1; x<cols-1; x++)
		{
			// Is it a local maximum
			int is_y_max = (sobel[y][x] > sobel[y-1][x] && sobel[y][x]>=sobel[y+1][x]);
			int is_x_max = (sobel[y][x] > sobel[y][x-1] && sobel[y][x]>=sobel[y][x+1]);
			if (is_y_max || is_x_max)
				nonmax[y][x] = sobel[y][x];
			else
				nonmax[y][x] = 0;
		}
	}
	
	sprintf(str, "out/%s_5_nonmax.png", outpre);
	SaveGrayPng(nonmax, str, rows, cols);
	
	//-----------------
	// Final Threshold
	//-----------------
	byte **edges = malloc2d(rows, cols);    // note: *this* initializes to zero!
	
	for (y=0; y<rows; y++) {
		for (x=0; x<cols; x++) {
			if (nonmax[y][x] > canny_thresh)
				edges[y][x] = 255;
			else
				edges[y][x] = 0;
		}
	}
	
	sprintf(str, "out/%s_6_edges.png", outpre);
	SaveGrayPng(edges, str, rows, cols);
	
  
    int count = 0;
	DisjointSet **edges1 = (DisjointSet **)malloc(rows * sizeof(DisjointSet *));
	for(int i = 0; i < rows; i++){
    edges1[i] = (DisjointSet*)malloc(cols * sizeof(DisjointSet));
}
	
	//Initialization of 2D array of structure
	for(int i=0;i<rows;i++)
	{
		for(int j=0;j<cols;j++)
		{
			if(edges[i][j] == 255){
           edges1[i][j].r = 0;
		   edges1[i][j].g = 0;
		   edges1[i][j].b = 0;
		   edges1[i][j].parent = &edges1[i][j];
		   edges1[i][j].rank = 0;
		   edges1[i][j].x = j+79;
		   edges1[i][j].y = i+79;
		   edges1[i][j].label = 0;
		   edges1[i][j].gaussian_r = 0;
		   edges1[i][j].gaussian_g = 0;
		   edges1[i][j].gaussian_b = 0;
		   edges1[i][j].gaussian_x = 0;
		   edges1[i][j].gaussian_y = 0;
			}
		}
	}
	
	int threshold1 = 0,d;
	byte **new_image = malloc2d(rows+158, cols+158);    // Added for padding

	for(int w=0;w<rows+158;w++)
	{
		for(int h=0;h<cols+158;h++)
		{
			new_image[w][h] = 0;
		}
	}
	for(int w=0;w<rows;w++)
	{
		for(int h=0;h<cols;h++)
		{
			new_image[w+79][h+79] = edges[w][h];
		}
	}


//Hyperparameters
 int size, new_i, new_j, sigma;
 size = gridSize;  // An odd size grid is considered so that union by rank is done between the midpoint and other edges for each grid
 new_i = iStride;
 new_j = jStride;
 sigma = sigmaVal;


//Code to call Union by Rank and hence Path Compression
	for(int i=0;i<rows;i=i+new_i)
	  {
		for(int j=0;j<cols;j=j+new_j)
		{
			if(new_image[i+79][j+79] == 255){
			for(int a=i-size;a<=i+size;a++)
			{
				for(int b=j-size;b<=j+size;b++)
				{
                     if(new_image[a+79][b+79] == 255)
					 {
						 double hypothesis1 = sqrt(pow((edges1[i][j].x - edges1[a][b].x),2) + pow((edges1[i][j].y - edges1[a][b].y),2));   // Euclidean Distance
						 if(hypothesis1 < 55){
						 DisjointSetUnion(&edges1[i][j],&edges1[a][b]);
						 }
					 }
				}
			}
		}
	  }
  }
   byte ***latest = (byte***)malloc3d(rows+158, cols+158, 3); 
	int c=0,value = 0;
	double x_coord, y_coord, wtavg;
	int x_coord1,y_coord1,cc;
	double gaussian;
	struct Node* head = NULL;
	struct Node* temp = NULL;
	struct Node* new_node = NULL;
	struct Node* temp1 = NULL;

	struct Node1* head1 = NULL;
	struct Node1* temp2 = NULL;
	struct Node1* new_node1 = NULL;
	struct Node1* temp3 = NULL;
    struct Node1* temp4 = NULL;
	struct Node1* temp5 = NULL;
	struct Node1* temp6 = NULL;
   

    // Code for Connected Components Colouring
	int r1 = rand() % 255;
    int g1 = rand() % 255;
	int b1 = rand() % 255;
	DisjointSet *obj,*obj1;
	int color = 255;
	for (y=0; y<rows; y++) {
		for (x=0; x<cols; x++) {
          if(edges[y][x] == 255){
            obj = DisjointSetFindRoot(&edges1[y][x]);
            if(edges1[y][x].r == 0 && edges1[y][x].g ==0 && edges1[y][x].b == 0 && edges1[y][x].label == 0)
			{
		    // Linked list implementation for labeling and colouring of components
			if(head1 == NULL)
					   {
						   head1 = (struct Node1*)malloc(sizeof(struct Node1));
						   head1->x_coordinate = edges1[y][x].x;
						   head1->y_coordinate = edges1[y][x].y;
						   head1->label = ++value;;
						   edges1[y][x].label = head1->label;
						   obj->label = head1->label;
						   head1->next1 = NULL;
						obj->r = r1;   
				       edges1[y][x].r = obj->r;
				       latest[y+79][x+79][0] = edges1[y][x].r;
					   obj->g = g1;
	                   edges1[y][x].g = obj->g;
					   latest[y+79][x+79][1] = edges1[y][x].g;
					   obj->b = b1;
	                   edges1[y][x].b = obj->b;
				       latest[y+79][x+79][2] = edges1[y][x].b;
					   }
					   else{
					   temp2 = (struct Node1*)malloc(sizeof(struct Node1));
                       temp2 = head1;
					   int count =0;
					   while(temp2->next1 != NULL){
						   if(edges1[y][x].x != temp2->x_coordinate && edges1[y][x].y != temp2->y_coordinate){
							   obj1 = DisjointSetFindRoot(&edges1[temp2->y_coordinate-79][temp2->x_coordinate-79]);
							   if(obj->x == obj1->x && obj->y == obj1->y ){
                                  
                                   count = 1;
								   break;
							   }

						   }
						   temp2 = temp2->next1;
					   }
					   
					   if(count == 1){
					   new_node1 = (struct Node1*)malloc(sizeof(struct Node1));
                       new_node1->x_coordinate = edges1[y][x].x;
			           new_node1->y_coordinate = edges1[y][x].y;
					   new_node1->label = obj->label;
					   new_node1->next1 = NULL;
					   new_node1->next1 = temp2->next1;
					   temp2->next1 = new_node1;
					   edges1[y][x].label = obj->label;
				       edges1[y][x].r = obj->r;
				       latest[y+79][x+79][0] = edges1[y][x].r;
	                   edges1[y][x].g = obj->g;
				       latest[y+79][x+79][1] = edges1[y][x].g;
					   edges1[y][x].b = obj->b;
				       latest[y+79][x+79][2] = edges1[y][x].b;
					   }
					   else{
						   
							   obj1 = DisjointSetFindRoot(&edges1[temp2->y_coordinate-79][temp2->x_coordinate-79]);
							   if(obj->x == obj1->x && obj->y == obj1->y && obj1->label != 0 && obj1->r != 0 && obj1->g != 0 && obj1->b != 0 
							   && obj->label != 0 && obj->r != 0 && obj->g != 0 && obj->b != 0){
								  
								   new_node1 = (struct Node1*)malloc(sizeof(struct Node1));
                       new_node1->x_coordinate = edges1[y][x].x;
			           new_node1->y_coordinate = edges1[y][x].y;
					   new_node1->label = obj->label;
					   new_node1->next1 =NULL;
					   temp2->next1 = new_node1;
					   edges1[y][x].label = obj->label;
				       edges1[y][x].r = obj->r;
				       latest[y+79][x+79][0] = edges1[y][x].r;
	                   edges1[y][x].g = obj->g;
				       latest[y+79][x+79][1] = edges1[y][x].g;
					   edges1[y][x].b = obj->b;
				       latest[y+79][x+79][2] = edges1[y][x].b;
							   }
					   
					  else{
						  
					   obj1 = DisjointSetFindRoot(&edges1[temp2->y_coordinate-79][temp2->x_coordinate-79]);
					   new_node1 = (struct Node1*)malloc(sizeof(struct Node1));
                       new_node1->x_coordinate = edges1[y][x].x;
			           new_node1->y_coordinate = edges1[y][x].y;
					   new_node1->label = ++value;
					   new_node1->next1 =NULL;
					   temp2->next1 = new_node1;

					   edges1[y][x].label = new_node1->label;
					   r1 = rand() % 255;
                       g1 = rand() % 255;
	                   b1 = rand() % 255;
				       edges1[y][x].r = r1;
				       latest[y+79][x+79][0] = edges1[y][x].r;
	                   edges1[y][x].g = g1;
				       latest[y+79][x+79][1] = edges1[y][x].g;
					   edges1[y][x].b = b1;
				       latest[y+79][x+79][2] = edges1[y][x].b;
                       if(obj1->label == 0 && obj1->r == 0 && obj1->g == 0 && obj1->b == 0){
						   obj1->label = new_node1->label;
						obj1->r = r1;   
					   obj1->g = g1;
					   obj1->b = b1;
					   }
					   }
					   
					   }
			}
		  }}}
			}
    sprintf(str, "out/%s_7_color_edges.png", outpre);
	SaveRgbPng(latest, str, rows+158, cols+158);


			// Code for Gaussian Kernel Smoothing
                      temp5 = (struct Node1*)malloc(sizeof(struct Node1));
					  
                       temp5 = head1;
					   
					   while(temp5->next1 != NULL){
						   x_coord = 0.0,y_coord= 0.0,wtavg = 0.0;
			               x_coord1 = 0,y_coord1 = 0,cc=0;
			               gaussian =0.0; 
						   temp6 = (struct Node1*)malloc(sizeof(struct Node1));
						   temp6 = head1;
						  while(temp6->next1 != NULL){
						   
		if(temp5->label == temp6->label)
			{
				double hypothesis = pow((temp5->x_coordinate - temp6->x_coordinate),2) + pow((temp5->y_coordinate - temp6->y_coordinate),2);
                gaussian = exp(-(hypothesis/(2*pow(sigma,2))))/(sqrt(2*(22/7))*sigma);
			    x_coord = x_coord + gaussian*temp6->x_coordinate; 
					   y_coord = y_coord + gaussian*temp6->y_coordinate; 
                       wtavg = wtavg + gaussian;
					   cc = 1;
			}
			 temp6 = temp6->next1;
						  }
			if(cc == 1){
			
	              x_coord1 = (int)(x_coord/wtavg);
				  y_coord1 = (int)(y_coord/wtavg);
		          edges1[temp5->y_coordinate-79][temp5->x_coordinate-79].gaussian_x = x_coord1;
				  edges1[temp5->y_coordinate-79][temp5->x_coordinate-79].gaussian_y = y_coord1;
				  if(head == NULL)
					   {
						  
						   head = (struct Node*)malloc(sizeof(struct Node));
						   head->gaussian_x = x_coord1;
						   head->gaussian_y = y_coord1;
						   head->next = NULL;
					   }
					   else{
					   temp = (struct Node*)malloc(sizeof(struct Node));
                       temp = head;
					   while(temp->next != NULL){
						   temp = temp->next;
					   }
					   new_node = (struct Node*)malloc(sizeof(struct Node));
                       new_node->gaussian_x = x_coord1;
			           new_node->gaussian_y = y_coord1;;
					   new_node->next =NULL;
					   temp->next = new_node;
					   
					   }
		  }
				 temp5 = temp5->next1;
					   } 
			
	temp1 = (struct Node*)malloc(sizeof(struct Node));
	temp1 = head;
	while(temp1 != 	NULL)
	{
		latest[temp1->gaussian_y][temp1->gaussian_x][0] = 255;
	    latest[temp1->gaussian_y][temp1->gaussian_x][1] = 255;     // Gaussian points plotted with white colour
		latest[temp1->gaussian_y][temp1->gaussian_x][2] = 255;
		temp1 = temp1->next;
	}
	sprintf(str, "out/%s_8_smooth_edges.png", outpre);
	SaveRgbPng(latest, str, rows+158, cols+158);

}

int main()
{
	system("mkdir out");
	
	//
	// Main simply calls the run function 
	//  with different parameters for each image
	//
	run("puppy.jpg", "puppy", 45, 25, 3, 2, 2, 9);
	run("pentagon.png", "pentagon", 45, 10, 25, 1, 1, 8);
	run("tiger.jpg", "tiger", 45, 10, 3, 2, 2, 8);
	run("houses.jpg", "houses", 45, 10, 3, 2, 3, 8);
	
	printf("Done!\n");
}
