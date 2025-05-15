import React, { useState } from 'react';
import {
  ThemeProvider,
  createTheme,
  CssBaseline,
  Box,
  AppBar,
  Toolbar,
  Typography,
  Drawer,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  Divider,
  IconButton,
  Container,
  Paper
} from '@mui/material';
import {
  Dashboard as DashboardIcon,
  DataObject as DataIcon,
  Memory as MemoryIcon,
  Assessment as AssessmentIcon,
  Add as AddIcon,
  Menu as MenuIcon,
  BrightnessHigh as LightIcon,
  Brightness4 as DarkIcon
} from '@mui/icons-material';

import ExperimentsList from './components/ExperimentsList';
import ExperimentDetail from './components/ExperimentDetail';
import CreateExperiment from './components/CreateExperiment';
import SystemResources from './components/SystemResources';

const App = () => {
  const [darkMode, setDarkMode] = useState(false);
  const [menuOpen, setMenuOpen] = useState(false);
  const [currentView, setCurrentView] = useState('experiments');
  const [selectedExperimentId, setSelectedExperimentId] = useState(null);

  const theme = createTheme({
    palette: {
      mode: darkMode ? 'dark' : 'light',
      primary: {
        main: '#3f51b5',
      },
      secondary: {
        main: '#f50057',
      },
    },
  });

  const handleSelectExperiment = (experimentId) => {
    setSelectedExperimentId(experimentId);
    setCurrentView('experiment-detail');
  };

  const handleExperimentCreated = (experiment) => {
    setSelectedExperimentId(experiment.id);
    setCurrentView('experiment-detail');
  };

  const handleBackToExperiments = () => {
    setSelectedExperimentId(null);
    setCurrentView('experiments');
  };

  const renderContent = () => {
    switch (currentView) {
      case 'experiment-detail':
        return (
          <ExperimentDetail 
            experimentId={selectedExperimentId} 
            onBack={handleBackToExperiments} 
          />
        );
      case 'create-experiment':
        return (
          <CreateExperiment 
            onExperimentCreated={handleExperimentCreated} 
          />
        );
      case 'system-resources':
        return <SystemResources />;
      case 'experiments':
      default:
        return (
          <ExperimentsList 
            onSelectExperiment={handleSelectExperiment} 
          />
        );
    }
  };

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Box sx={{ display: 'flex' }}>
        {/* App Bar */}
        <AppBar position="fixed">
          <Toolbar>
            <IconButton
              color="inherit"
              edge="start"
              onClick={() => setMenuOpen(true)}
              sx={{ mr: 2 }}
            >
              <MenuIcon />
            </IconButton>
            <Typography variant="h6" component="div" sx={{ flexGrow: 1 }}>
              RLHF Engineering System
            </Typography>
            <IconButton color="inherit" onClick={() => setDarkMode(!darkMode)}>
              {darkMode ? <LightIcon /> : <DarkIcon />}
            </IconButton>
          </Toolbar>
        </AppBar>

        {/* Drawer */}
        <Drawer
          open={menuOpen}
          onClose={() => setMenuOpen(false)}
        >
          <Box
            sx={{ width: 250 }}
            role="presentation"
            onClick={() => setMenuOpen(false)}
          >
            <List>
              <ListItem 
                button 
                onClick={() => setCurrentView('experiments')}
                selected={currentView === 'experiments'}
              >
                <ListItemIcon><DashboardIcon /></ListItemIcon>
                <ListItemText primary="Experiments" />
              </ListItem>
              <ListItem 
                button 
                onClick={() => setCurrentView('create-experiment')}
                selected={currentView === 'create-experiment'}
              >
                <ListItemIcon><AddIcon /></ListItemIcon>
                <ListItemText primary="Create Experiment" />
              </ListItem>
              <ListItem 
                button 
                onClick={() => setCurrentView('system-resources')}
                selected={currentView === 'system-resources'}
              >
                <ListItemIcon><MemoryIcon /></ListItemIcon>
                <ListItemText primary="System Resources" />
              </ListItem>
            </List>
            <Divider />
            <List>
              <ListItem button disabled>
                <ListItemIcon><DataIcon /></ListItemIcon>
                <ListItemText primary="Models" />
              </ListItem>
              <ListItem button disabled>
                <ListItemIcon><AssessmentIcon /></ListItemIcon>
                <ListItemText primary="Analytics" />
              </ListItem>
            </List>
          </Box>
        </Drawer>

        {/* Main Content */}
        <Box
          component="main"
          sx={{
            flexGrow: 1,
            p: 3,
            width: '100%',
            mt: 8,
          }}
        >
          <Container maxWidth="xl">
            {renderContent()}
          </Container>
        </Box>
      </Box>
    </ThemeProvider>
  );
};

export default App;