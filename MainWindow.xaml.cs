﻿using MURDOC.ViewModel;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;
using System.Windows.Shapes;

namespace MURDOC
{
    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class MainWindow : Window
    {
        public MainWindow()
        {
            InitializeComponent();

            // Create an instance of MainWindowViewModel and set it as the DataContext
            DataContext = new MainWindowViewModel();
        }

        private void LocalizationImage_MouseEnter(object sender, System.Windows.Input.MouseEventArgs e)
        {
            var imageItem = sender as System.Windows.Controls.Image;
            if (imageItem != null)
            {
                var viewModel = DataContext as MainWindowViewModel;
                if (viewModel != null)
                {
                    viewModel.HandlePreviewImageChanged(imageItem.Source.ToString());
                }
            }   
        }
    }
}
