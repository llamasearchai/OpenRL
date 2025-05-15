import React, { useState, useEffect } from 'react';
import { invoke } from '@tauri-apps/api/tauri';
import {
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  Button,
  Typography,
  Box,
  Chip,
  CircularProgress
} from '@mui/material';

const ExperimentsList = ({ onSelectExperiment }) => {
  const [experiments, setExperiments] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    loadExperiments();
  }, []);

  const loadExperiments = async () => {
    setLoading(true);
    try {
      const result = await invoke('get_experiments');
      setExperiments(result);
      setError(null);
    } catch (err) {
      console.error('Failed to load experiments:', err);
      setError('Failed to load experiments: ' + err);
    } finally {
      setLoading(false);
    }
  };

  const handleStartTraining = async (experimentId) => {
    try {
      await invoke('start_training', { experimentId });
      loadExperiments(); // Refresh the list
    } catch (err) {
      console.error('Failed to start training:', err);
      setError('Failed to start training: ' + err);
    }
  };

  const handleStopTraining = async (experimentId) => {
    try {
      await invoke('stop_training', { experimentId });
      loadExperiments(); // Refresh the list
    } catch (err) {
      console.error('Failed to stop training:', err);
      setError('Failed to stop training: ' + err);
    }
  };

  const getStatusColor = (status) => {
    switch (status) {
      case 'running':
        return 'success';
      case 'completed':
        return 'primary';
      case 'stopped':
        return 'warning';
      case 'failed':
        return 'error';
      default:
        return 'default';
    }
  };

  if (loading) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" p={4}>
        <CircularProgress />
      </Box>
    );
  }

  if (error) {
    return (
      <Box p={2}>
        <Typography color="error">{error}</Typography>
        <Button variant="contained" onClick={loadExperiments} sx={{ mt: 2 }}>
          Retry
        </Button>
      </Box>
    );
  }

  return (
    <Box p={2}>
      <Box display="flex" justifyContent="space-between" alignItems="center" mb={2}>
        <Typography variant="h5">RLHF Experiments</Typography>
        <Button variant="contained" color="primary" onClick={loadExperiments}>
          Refresh
        </Button>
      </Box>
      
      <TableContainer component={Paper}>
        <Table>
          <TableHead>
            <TableRow>
              <TableCell>Name</TableCell>
              <TableCell>Algorithm</TableCell>
              <TableCell>Status</TableCell>
              <TableCell>Start Time</TableCell>
              <TableCell>Metrics</TableCell>
              <TableCell>Actions</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {experiments.length === 0 ? (
              <TableRow>
                <TableCell colSpan={6} align="center">
                  <Typography variant="body1">No experiments found</Typography>
                </TableCell>
              </TableRow>
            ) : (
              experiments.map((experiment) => (
                <TableRow key={experiment.id}>
                  <TableCell>
                    <Typography
                      variant="body1"
                      sx={{ cursor: 'pointer', '&:hover': { textDecoration: 'underline' } }}
                      onClick={() => onSelectExperiment(experiment.id)}
                    >
                      {experiment.name}
                    </Typography>
                  </TableCell>
                  <TableCell>{experiment.algorithm}</TableCell>
                  <TableCell>
                    <Chip 
                      label={experiment.status} 
                      color={getStatusColor(experiment.status)} 
                      size="small"
                    />
                  </TableCell>
                  <TableCell>{new Date(experiment.start_time).toLocaleString()}</TableCell>
                  <TableCell>
                    {experiment.metrics && (
                      <Box>
                        {Object.entries(experiment.metrics).map(([key, value]) => (
                          <Typography variant="body2" key={key}>
                            {key}: {value.toFixed(4)}
                          </Typography>
                        ))}
                      </Box>
                    )}
                  </TableCell>
                  <TableCell>
                    {experiment.status === 'running' ? (
                      <Button 
                        variant="outlined" 
                        color="error" 
                        size="small"
                        onClick={() => handleStopTraining(experiment.id)}
                      >
                        Stop
                      </Button>
                    ) : experiment.status !== 'completed' ? (
                      <Button 
                        variant="outlined" 
                        color="success" 
                        size="small"
                        onClick={() => handleStartTraining(experiment.id)}
                      >
                        Start
                      </Button>
                    ) : null}
                  </TableCell>
                </TableRow>
              ))
            )}
          </TableBody>
        </Table>
      </TableContainer>
    </Box>
  );
};