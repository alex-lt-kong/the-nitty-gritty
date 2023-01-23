namespace faster_csharp
{
    partial class FormMain
    {
        /// <summary>
        ///  Required designer variable.
        /// </summary>
        private System.ComponentModel.IContainer components = null;

        /// <summary>
        ///  Clean up any resources being used.
        /// </summary>
        /// <param name="disposing">true if managed resources should be disposed; otherwise, false.</param>
        protected override void Dispose(bool disposing)
        {
            if (disposing && (components != null))
            {
                components.Dispose();
            }
            base.Dispose(disposing);
        }

        #region Windows Form Designer generated code

        /// <summary>
        ///  Required method for Designer support - do not modify
        ///  the contents of this method with the code editor.
        /// </summary>
        private void InitializeComponent()
        {
            this.buttonGCvsLoop = new System.Windows.Forms.Button();
            this.textBoxOutput = new System.Windows.Forms.TextBox();
            this.buttonStaticVSDynamicArraies = new System.Windows.Forms.Button();
            this.buttonArrayVSArrayPool = new System.Windows.Forms.Button();
            this.buttonStructVSClass = new System.Windows.Forms.Button();
            this.buttonTryVSNoTry = new System.Windows.Forms.Button();
            this.buttonFinalizerVSNoFinalizer = new System.Windows.Forms.Button();
            this.buttonStringVSStringBuilder = new System.Windows.Forms.Button();
            this.buttonSOHvsLOH = new System.Windows.Forms.Button();
            this.buttonHashtableVSDictionary = new System.Windows.Forms.Button();
            this.buttonDivideVSMultiplyByReciprocal = new System.Windows.Forms.Button();
            this.button1 = new System.Windows.Forms.Button();
            this.SuspendLayout();
            // 
            // buttonGCvsLoop
            // 
            this.buttonGCvsLoop.Location = new System.Drawing.Point(12, 12);
            this.buttonGCvsLoop.Name = "buttonGCvsLoop";
            this.buttonGCvsLoop.Size = new System.Drawing.Size(188, 40);
            this.buttonGCvsLoop.TabIndex = 0;
            this.buttonGCvsLoop.Text = "Garbage Collectioin vs Loop";
            this.buttonGCvsLoop.UseVisualStyleBackColor = true;
            this.buttonGCvsLoop.Click += new System.EventHandler(this.buttonGCvsLoop_Click);
            // 
            // textBoxOutput
            // 
            this.textBoxOutput.Anchor = ((System.Windows.Forms.AnchorStyles)((((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Bottom) 
            | System.Windows.Forms.AnchorStyles.Left) 
            | System.Windows.Forms.AnchorStyles.Right)));
            this.textBoxOutput.BackColor = System.Drawing.Color.White;
            this.textBoxOutput.Location = new System.Drawing.Point(206, 12);
            this.textBoxOutput.Multiline = true;
            this.textBoxOutput.Name = "textBoxOutput";
            this.textBoxOutput.ReadOnly = true;
            this.textBoxOutput.ScrollBars = System.Windows.Forms.ScrollBars.Vertical;
            this.textBoxOutput.Size = new System.Drawing.Size(387, 514);
            this.textBoxOutput.TabIndex = 1;
            // 
            // buttonStaticVSDynamicArraies
            // 
            this.buttonStaticVSDynamicArraies.Location = new System.Drawing.Point(12, 58);
            this.buttonStaticVSDynamicArraies.Name = "buttonStaticVSDynamicArraies";
            this.buttonStaticVSDynamicArraies.Size = new System.Drawing.Size(188, 40);
            this.buttonStaticVSDynamicArraies.TabIndex = 2;
            this.buttonStaticVSDynamicArraies.Text = "Static vs dynamic arraies";
            this.buttonStaticVSDynamicArraies.UseVisualStyleBackColor = true;
            this.buttonStaticVSDynamicArraies.Click += new System.EventHandler(this.buttonStaticVSDynamicArraies_Click);
            // 
            // buttonArrayVSArrayPool
            // 
            this.buttonArrayVSArrayPool.Location = new System.Drawing.Point(12, 104);
            this.buttonArrayVSArrayPool.Name = "buttonArrayVSArrayPool";
            this.buttonArrayVSArrayPool.Size = new System.Drawing.Size(188, 40);
            this.buttonArrayVSArrayPool.TabIndex = 3;
            this.buttonArrayVSArrayPool.Text = "Array vs ArrayPool";
            this.buttonArrayVSArrayPool.UseVisualStyleBackColor = true;
            this.buttonArrayVSArrayPool.Click += new System.EventHandler(this.buttonArrayVSArrayPool_Click);
            // 
            // buttonStructVSClass
            // 
            this.buttonStructVSClass.Location = new System.Drawing.Point(12, 150);
            this.buttonStructVSClass.Name = "buttonStructVSClass";
            this.buttonStructVSClass.Size = new System.Drawing.Size(188, 40);
            this.buttonStructVSClass.TabIndex = 4;
            this.buttonStructVSClass.Text = "Class vs Struct vs Dictionary";
            this.buttonStructVSClass.UseVisualStyleBackColor = true;
            this.buttonStructVSClass.Click += new System.EventHandler(this.buttonStructVSClass_Click);
            // 
            // buttonTryVSNoTry
            // 
            this.buttonTryVSNoTry.Location = new System.Drawing.Point(12, 196);
            this.buttonTryVSNoTry.Name = "buttonTryVSNoTry";
            this.buttonTryVSNoTry.Size = new System.Drawing.Size(188, 40);
            this.buttonTryVSNoTry.TabIndex = 5;
            this.buttonTryVSNoTry.Text = "Try vs No try";
            this.buttonTryVSNoTry.UseVisualStyleBackColor = true;
            this.buttonTryVSNoTry.Click += new System.EventHandler(this.buttonTryVSNoTry_Click);
            // 
            // buttonFinalizerVSNoFinalizer
            // 
            this.buttonFinalizerVSNoFinalizer.Location = new System.Drawing.Point(12, 242);
            this.buttonFinalizerVSNoFinalizer.Name = "buttonFinalizerVSNoFinalizer";
            this.buttonFinalizerVSNoFinalizer.Size = new System.Drawing.Size(188, 40);
            this.buttonFinalizerVSNoFinalizer.TabIndex = 6;
            this.buttonFinalizerVSNoFinalizer.Text = "Finalizer vs No finalizer";
            this.buttonFinalizerVSNoFinalizer.UseVisualStyleBackColor = true;
            this.buttonFinalizerVSNoFinalizer.Click += new System.EventHandler(this.buttonFinalizerVSNoFinalizer_Click);
            // 
            // buttonStringVSStringBuilder
            // 
            this.buttonStringVSStringBuilder.Location = new System.Drawing.Point(12, 288);
            this.buttonStringVSStringBuilder.Name = "buttonStringVSStringBuilder";
            this.buttonStringVSStringBuilder.Size = new System.Drawing.Size(188, 40);
            this.buttonStringVSStringBuilder.TabIndex = 7;
            this.buttonStringVSStringBuilder.Text = "String vs StringBuilder";
            this.buttonStringVSStringBuilder.UseVisualStyleBackColor = true;
            this.buttonStringVSStringBuilder.Click += new System.EventHandler(this.buttonStringVSStringBuilder_Click);
            // 
            // buttonSOHvsLOH
            // 
            this.buttonSOHvsLOH.Location = new System.Drawing.Point(12, 334);
            this.buttonSOHvsLOH.Name = "buttonSOHvsLOH";
            this.buttonSOHvsLOH.Size = new System.Drawing.Size(188, 40);
            this.buttonSOHvsLOH.TabIndex = 8;
            this.buttonSOHvsLOH.Text = "SOH vs LOH";
            this.buttonSOHvsLOH.UseVisualStyleBackColor = true;
            this.buttonSOHvsLOH.Click += new System.EventHandler(this.buttonSOHvsLOH_Click);
            // 
            // buttonHashtableVSDictionary
            // 
            this.buttonHashtableVSDictionary.Location = new System.Drawing.Point(12, 380);
            this.buttonHashtableVSDictionary.Name = "buttonHashtableVSDictionary";
            this.buttonHashtableVSDictionary.Size = new System.Drawing.Size(188, 40);
            this.buttonHashtableVSDictionary.TabIndex = 9;
            this.buttonHashtableVSDictionary.Text = "Hashtable vs Dictionary";
            this.buttonHashtableVSDictionary.UseVisualStyleBackColor = true;
            this.buttonHashtableVSDictionary.Click += new System.EventHandler(this.buttonHashtableVSDictionary_Click);
            // 
            // buttonDivideVSMultiplyByReciprocal
            // 
            this.buttonDivideVSMultiplyByReciprocal.Location = new System.Drawing.Point(12, 426);
            this.buttonDivideVSMultiplyByReciprocal.Name = "buttonDivideVSMultiplyByReciprocal";
            this.buttonDivideVSMultiplyByReciprocal.Size = new System.Drawing.Size(188, 40);
            this.buttonDivideVSMultiplyByReciprocal.TabIndex = 10;
            this.buttonDivideVSMultiplyByReciprocal.Text = "Divide vs Multiply by Reciprocal ";
            this.buttonDivideVSMultiplyByReciprocal.UseVisualStyleBackColor = true;
            this.buttonDivideVSMultiplyByReciprocal.Click += new System.EventHandler(this.buttonDivideVSMultiplyByReciprocal_Click);
            // 
            // button1
            // 
            this.button1.Location = new System.Drawing.Point(12, 472);
            this.button1.Name = "button1";
            this.button1.Size = new System.Drawing.Size(188, 40);
            this.button1.TabIndex = 11;
            this.button1.Text = "gRPC vs RESTful API";
            this.button1.UseVisualStyleBackColor = true;
            this.button1.Click += new System.EventHandler(this.button1_Click);
            // 
            // FormMain
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(7F, 15F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(605, 538);
            this.Controls.Add(this.button1);
            this.Controls.Add(this.buttonDivideVSMultiplyByReciprocal);
            this.Controls.Add(this.buttonHashtableVSDictionary);
            this.Controls.Add(this.buttonSOHvsLOH);
            this.Controls.Add(this.buttonStringVSStringBuilder);
            this.Controls.Add(this.buttonFinalizerVSNoFinalizer);
            this.Controls.Add(this.buttonTryVSNoTry);
            this.Controls.Add(this.buttonStructVSClass);
            this.Controls.Add(this.buttonArrayVSArrayPool);
            this.Controls.Add(this.buttonStaticVSDynamicArraies);
            this.Controls.Add(this.textBoxOutput);
            this.Controls.Add(this.buttonGCvsLoop);
            this.Name = "FormMain";
            this.Text = "MainForm";
            this.ResumeLayout(false);
            this.PerformLayout();

        }

        #endregion

        private Button buttonGCvsLoop;
        private TextBox textBoxOutput;
        private Button buttonStaticVSDynamicArraies;
        private Button buttonArrayVSArrayPool;
        private Button buttonStructVSClass;
        private Button buttonTryVSNoTry;
        private Button buttonFinalizerVSNoFinalizer;
        private Button buttonStringVSStringBuilder;
        private Button buttonSOHvsLOH;
        private Button buttonHashtableVSDictionary;
        private Button buttonDivideVSMultiplyByReciprocal;
        private Button button1;
    }
}