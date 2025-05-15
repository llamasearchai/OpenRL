import React, { useState, useEffect } from 'react';
import { invoke } from '@tauri-apps/api/tauri';
import {
  Box,
  Typography,
  Paper,
  Grid,
  Button,
  CircularProgress,
  Tabs,
  Tab,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Chip
} from '@mui/material';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

const ExperimentDetail = ({ experimentId, onBack }) => {
  const [experiment, setExperiment] = useState(null);
  const [lossHistory, setLossHistory] = useState([]);
  const [resourceMetrics, setResourceMetrics] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [activeTab, setActiveTab] = useState(0);

  useEffect(() => {
    loadExperimentDetails();
    const interval = setInterval(() => {
      if (experiment?.status === 'running') {
        loadExperimentDetails();
        loadLossHistory();
        loadResourceMetrics('5min');
      }
    }, 10000); // Update every 10 seconds for running experiments
    
    return () => clearInterval(interval);
  }, [experimentId, experiment?.status]);
  
  const loadExperimentDetails = async () => {
    setLoading(true);
    try {
      const result = await invoke('get_experiment_details', { experimentId });
      setExperiment(result);
      loadLossHistory();
      loadResourceMetrics('5min');
      setError(null);
    } catch (err) {
      console.error('Failed to load experiment details:', err);
      setError('Failed to load experiment details: ' + err);
    } finally {
      setLoading(false);
    }
  };
  
  const loadLossHistory = async () => {
    try {
      const result = await invoke('get_loss_history', { experimentId });
      setLossHistory(result.map(item => ({
        step: item.step,
        loss: item.value,
        timestamp: new Date(item.timestamp).toLocaleString()
      })));
    } catch (err) {
      console.error('Failed to load loss history:', err);
    }
  };
  
  const loadResourceMetrics = async (timeframe) => {
    try {
      const result = await invoke('get_resource_metrics', { timeframe });
      setResourceMetrics(result);
    } catch (err) {
      console.error('Failed to load resource metrics:', err);
    }
  };

  const handleTabChange = (event, newValue) => {
    setActiveTab(newValue);
  };

  const handleStartTraining = async () => {
    try {
      await invoke('start_training', { experimentId });
      loadExperimentDetails();
    } catch (err) {
      setError('Failed to start training: ' + err);
    }
  };
  
  const handleStopTraining = async () => {
    try {
      await invoke('stop_training', { experimentId });
      loadExperimentDetails();
    } catch (err) {
      setError('Failed to stop training: ' + err);
    }
  };

  if (loading && !experiment) {
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
        <Button variant="contained" onClick={loadExperimentDetails} sx={{ mt: 2 }}>
          Retry
        </Button>
        <Button variant="outlined" onClick={onBack} sx={{ mt: 2, ml: 2 }}>
          Back to Experiments
        </Button>
      </Box>
    );
  }

  if (!experiment) {
    return (
      <Box p={2}>
        <Typography>No experiment found</Typography>
        <Button variant="outlined" onClick={onBack} sx={{ mt: 2 }}>
          Back to Experiments
        </Button>
      </Box>
    );
  }

  return (
    <Box p={2}>
      <Box display="flex" justifyContent="space-between" alignItems="center" mb={2}>
        <Box>
          <Button variant="outlined" onClick={onBack} sx={{ mr: 2 }}>
            Back
          </Button>
          <Typography variant="h5" component="span">
            {experiment.name}
          </Typography>
        </Box>
        <Box>
          {experiment.status === 'running' ? (
            <Button variant="contained" color="error" onClick={handleStopTraining}>
              Stop Training
            </Button>
          ) : experiment.status !== 'completed' ? (
            <Button variant="contained" color="primary" onClick={handleStartTraining}>
              Start Training
            </Button>
          ) : (
            <Chip label="Completed" color="success" />
          )}
        </Box>
      </Box>

      <Grid container spacing={2}>
        <Grid item xs={12} md={6}>
          <Paper sx={{ p: 2 }}>
            <Typography variant="h6" gutterBottom>Experiment Info</Typography>
            <Grid container spacing={1}>
              <Grid item xs={4}>
                <Typography variant="body2" color="textSecondary">Algorithm:</Typography>
              </Grid>
              <Grid item xs={8}>
                <Typography variant="body1">{experiment.algorithm}</Typography>
              </Grid>
              
              <Grid item xs={4}>
                <Typography variant="body2" color="textSecondary">Status:</Typography>
              </Grid>
              <Grid item xs={8}>
                <Chip 
                  label={experiment.status} 
                  color={
                    experiment.status === 'running' ? 'success' :
                    experiment.status === 'completed' ? 'primary' :
                    experiment.status === 'failed' ? 'error' : 'default'
                  } 
                  size="small"
                />
              </Grid>
              
              <Grid item xs={4}>
                <Typography variant="body2" color="textSecondary">Started:</Typography>
              </Grid>
              <Grid item xs={8}>
                <Typography variant="body1">{new Date(experiment.start_time).toLocaleString()}</Typography>
              </Grid>
              
              {experiment.end_time && (
                <>
                  <Grid item xs={4}>
                    <Typography variant="body2" color="textSecondary">Ended:</Typography>
                  </Grid>
                  <Grid item xs={8}>
                    <Typography variant="body1">{new Date(experiment.end_time).toLocaleString()}</Typography>
                  </Grid>
                </>
              )}
              
              <Grid item xs={4}>
                <Typography variant="body2" color="textSecondary">Duration:</Typography>
              </Grid>
              <Grid item xs={8}>
                <Typography variant="body1">
                  {experiment.duration_seconds 
                    ? `${Math.floor(experiment.duration_seconds / 60)}m ${experiment.duration_seconds % 60}s`
                    : 'In progress...'}
                </Typography>
              </Grid>
              
              <Grid item xs={4}>
                <Typography variant="body2" color="textSecondary">Model:</Typography>
              </Grid>
              <Grid item xs={8}>
                <Typography variant="body1">{experiment.model}</Typography>
              </Grid>
              
              <Grid item xs={4}>
                <Typography variant="body2" color="textSecondary">Dataset:</Typography>
              </Grid>
              <Grid item xs={8}>
                <Typography variant="body1">{experiment.dataset}</Typography>
              </Grid>
            </Grid>
          </Paper>
        </Grid>
        
        <Grid item xs={12} md={6}>
          <Paper sx={{ p: 2, height: '100%' }}>
            <Typography variant="h6" gutterBottom>Current Metrics</Typography>
            {experiment.metrics && Object.keys(experiment.metrics).length > 0 ? (
              <TableContainer>
                <Table size="small">
                  <TableHead>
                    <TableRow>
                      <TableCell>Metric</TableCell>
                      <TableCell align="right">Value</TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {Object.entries(experiment.metrics).map(([key, value]) => (
                      <TableRow key={key}>
                        <TableCell component="th" scope="row">
                          {key}
                        </TableCell>
                        <TableCell align="right">
                          {typeof value === 'number' ? value.toFixed(6) : value}
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </TableContainer>
            ) : (
              <Typography variant="body2" color="textSecondary">
                No metrics available yet
              </Typography>
            )}
          </Paper>
        </Grid>
        
        <Grid item xs={12}>
          <Paper sx={{ p: 2 }}>
            <Tabs value={activeTab} onChange={handleTabChange}>
              <Tab label="Training Progress" />
              <Tab label="System Resources" />
              <Tab label="Configuration" />
              <Tab label="Logs" />
            </Tabs>
            
            <Box sx={{ mt: 2 }}>
              {activeTab === 0 && (
                <Box>
                  <Typography variant="h6" gutterBottom>Training Loss</Typography>
                  {lossHistory.length > 0 ? (
                    <ResponsiveContainer width="100%" height={300}>
                      <LineChart data={lossHistory}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="step" />
                        <YAxis />
                        <Tooltip formatter={(value) => value.toFixed(6)} />
                        <Legend />
                        <Line type="monotone" dataKey="loss" stroke="#8884d8" activeDot={{ r: 8 }} />
                      </LineChart>
                    </ResponsiveContainer>
                  ) : (
                    <Typography>No training data available yet</Typography>
                  )}
                </Box>
              )}
              
              {activeTab === 1 && (
                <Box>
                  <Typography variant="h6" gutterBottom>System Resources</Typography>
                  {resourceMetrics ? (
                    <Grid container spacing={2}>
                      <Grid item xs={12} md={6}>
                        <ResponsiveContainer width="100%" height={200}>
                          <LineChart data={resourceMetrics.metrics}>
                            <CartesianGrid strokeDasharray="3 3" />
                            <XAxis dataKey="timestamp" tickFormatter={(tick) => ''} />
                            <YAxis />
                            <Tooltip 
                              formatter={(value) => `${value.toFixed(2)}%`}
                              labelFormatter={(label) => new Date(label * 1000).toLocaleTimeString()}
                            />
                            <Legend />
                            <Line type="monotone" dataKey="cpu_usage" name="CPU Usage (%)" stroke="#8884d8" />
                          </LineChart>
                        </ResponsiveContainer>
                      </Grid>
                      
                      <Grid item xs={12} md={6}>
                        <ResponsiveContainer width="100%" height={200}>
                          <LineChart data={resourceMetrics.metrics}>
                            <CartesianGrid strokeDasharray="3 3" />
                            <XAxis dataKey="timestamp" tickFormatter={(tick) => ''} />
                            <YAxis />
                            <Tooltip 
                              formatter={(value) => `${value.toFixed(2)} GB`}
                              labelFormatter={(label) => new Date(label * 1000).toLocaleTimeString()}
                            />
                            <Legend />
                            <Line type="monotone" dataKey="memory_usage" name="Memory Usage (GB)" stroke="#82ca9d" />
                          </LineChart>
                        </ResponsiveContainer>
                      </Grid>
                      
                      {resourceMetrics.metrics[0].gpu_usage && (
                        <Grid item xs={12} md={6}>
                          <ResponsiveContainer width="100%" height={200}>
                            <LineChart data={resourceMetrics.metrics}>
                              <CartesianGrid strokeDasharray="3 3" />
                              <XAxis dataKey="timestamp" tickFormatter={(tick) => ''} />
                              <YAxis />
                              <Tooltip 
                                formatter={(value) => `${value.toFixed(2)}%`}
                                labelFormatter={(label) => new Date(label * 1000).toLocaleTimeString()}
                              />
                              <Legend />
                              <Line type="monotone" dataKey="gpu_usage[0]" name="GPU Usage (%)" stroke="#ff7300" />
                            </LineChart>
                          </ResponsiveContainer>
                        </Grid>
                      )}
                      
                      {resourceMetrics.metrics[0].gpu_memory && (
                        <Grid item xs={12} md={6}>
                          <ResponsiveContainer width="100%" height={200}>
                            <LineChart data={resourceMetrics.metrics}>
                              <CartesianGrid strokeDasharray="3 3" />
                              <XAxis dataKey="timestamp" tickFormatter={(tick) => ''} />
                              <YAxis />
                              <Tooltip 
                                formatter={(value) => `${value.toFixed(2)} GB`}
                                labelFormatter={(label) => new Date(label * 1000).toLocaleTimeString()}
                              />
                              <Legend />
                              <Line type="monotone" dataKey="gpu_memory[0]" name="GPU Memory (GB)" stroke="#ff4500" />
                            </LineChart>
                          </ResponsiveContainer>
                        </Grid>
                      )}
                      
                      <Grid item xs={12} md={6}>
                        <Box sx={{ mt: 2 }}>
                          <Button variant="outlined" onClick={() => loadResourceMetrics('5min')}>Last 5 min</Button>
                          <Button variant="outlined" onClick={() => loadResourceMetrics('15min')} sx={{ ml: 1 }}>Last 15 min</Button>
                          <Button variant="outlined" onClick={() => loadResourceMetrics('1hr')} sx={{ ml: 1 }}>Last 1 hour</Button>
                          <Button variant="outlined" onClick={() => loadResourceMetrics('6hr')} sx={{ ml: 1 }}>Last 6 hours</Button>
                        </Box>
                      </Grid>
                    </Grid>
                  ) : (
                    <Typography>Loading resource metrics...</Typography>
                  )}
                </Box>
              )}
              
              {activeTab === 2 && (
                <Box>
                  <Typography variant="h6" gutterBottom>Configuration</Typography>
                  {experiment.config ? (
                    <Paper sx={{ p: 2, maxHeight: 400, overflow: 'auto' }}>
                      <pre style={{ margin: 0 }}>
                        {JSON.stringify(experiment.config, null, 2)}
                      </pre>
                    </Paper>
                  ) : (
                    <Typography>No configuration available</Typography>
                  )}
                </Box>
              )}
              
              {activeTab === 3 && (
                <Box>
                  <Typography variant="h6" gutterBottom>Training Logs</Typography>
                  <Paper sx={{ p: 2, maxHeight: 400, overflow: 'auto', fontFamily: 'monospace', fontSize: '0.875rem' }}>
                    {loading ? (
                      <CircularProgress size={24} />
                    ) : (
                      <Box component="div" sx={{ whiteSpace: 'pre-wrap' }}>
                        <Button 
                          variant="outlined" 
                          size="small" 
                          onClick={() => invoke('get_training_logs', { experimentId })}
                          sx={{ mb: 2 }}
                        >
                          Refresh Logs
                        </Button>
                        
                        {/* Logs would be loaded here */}
                        <Typography variant="body2" color="textSecondary">
                          Log output will appear here...
                        </Typography>
                      </Box>
                    )}
                  </Paper>
                </Box>
              )}
            </Box>
          </Paper>
        </Grid>
      </Grid>
    </Box>
  );
};

export default ExperimentDetail;