import React, { useState, useEffect } from 'react';
import { invoke } from '@tauri-apps/api/tauri';
import {
  Box,
  Typography,
  Paper,
  Grid,
  Button,
  CircularProgress,
} from '@mui/material';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

const SystemResources = () => {
  const [resourceMetrics, setResourceMetrics] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [timeframe, setTimeframe] = useState('5min');

  useEffect(() => {
    loadResourceMetrics();
    
    // Set up polling
    const interval = setInterval(() => {
      loadResourceMetrics();
    }, 5000); // Update every 5 seconds
    
    return () => clearInterval(interval);
  }, [timeframe]);
  
  const loadResourceMetrics = async () => {
    setLoading(true);
    try {
      const result = await invoke('get_resource_metrics', { timeframe });
      setResourceMetrics(result);
      setError(null);
    } catch (err) {
      console.error('Failed to load resource metrics:', err);
      setError('Failed to load resource metrics: ' + err);
    } finally {
      setLoading(false);
    }
  };

  if (loading && !resourceMetrics) {
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
        <Button variant="contained" onClick={loadResourceMetrics} sx={{ mt: 2 }}>
          Retry
        </Button>
      </Box>
    );
  }

  if (!resourceMetrics) {
    return (
      <Box p={2}>
        <Typography>No resource metrics available</Typography>
        <Button variant="contained" onClick={loadResourceMetrics} sx={{ mt: 2 }}>
          Load Metrics
        </Button>
      </Box>
    );
  }

  return (
    <Box p={2}>
      <Typography variant="h5" gutterBottom>System Resources</Typography>
      
      <Box sx={{ mb: 2 }}>
        <Button 
          variant={timeframe === '5min' ? 'contained' : 'outlined'} 
          onClick={() => setTimeframe('5min')}
          sx={{ mr: 1 }}
        >
          Last 5 min
        </Button>
        <Button 
          variant={timeframe === '15min' ? 'contained' : 'outlined'} 
          onClick={() => setTimeframe('15min')}
          sx={{ mr: 1 }}
        >
          Last 15 min
        </Button>
        <Button 
          variant={timeframe === '1hr' ? 'contained' : 'outlined'} 
          onClick={() => setTimeframe('1hr')}
          sx={{ mr: 1 }}
        >
          Last 1 hour
        </Button>
        <Button 
          variant={timeframe === '6hr' ? 'contained' : 'outlined'} 
          onClick={() => setTimeframe('6hr')}
        >
          Last 6 hours
        </Button>
      </Box>
      
      <Grid container spacing={2}>
        <Grid item xs={12} md={6}>
          <Paper sx={{ p: 2 }}>
            <Typography variant="h6" gutterBottom>CPU Usage</Typography>
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={resourceMetrics.metrics}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis 
                  dataKey="timestamp" 
                  tickFormatter={(tick) => new Date(tick * 1000).toLocaleTimeString()} 
                  minTickGap={30}
                />
                <YAxis domain={[0, 100]} />
                <Tooltip 
                  formatter={(value) => `${value.toFixed(2)}%`}
                  labelFormatter={(label) => new Date(label * 1000).toLocaleTimeString()}
                />
                <Legend />
                <Line type="monotone" dataKey="cpu_usage" name="CPU Usage (%)" stroke="#8884d8" />
              </LineChart>
            </ResponsiveContainer>
          </Paper>
        </Grid>
        
        <Grid item xs={12} md={6}>
          <Paper sx={{ p: 2 }}>
            <Typography variant="h6" gutterBottom>Memory Usage</Typography>
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={resourceMetrics.metrics}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis 
                  dataKey="timestamp" 
                  tickFormatter={(tick) => new Date(tick * 1000).toLocaleTimeString()}
                  minTickGap={30}
                />
                <YAxis />
                <Tooltip 
                  formatter={(value) => `${value.toFixed(2)} GB`}
                  labelFormatter={(label) => new Date(label * 1000).toLocaleTimeString()}
                />
                <Legend />
                <Line type="monotone" dataKey="memory_usage" name="Memory Used (GB)" stroke="#82ca9d" />
              </LineChart>
            </ResponsiveContainer>
          </Paper>
        </Grid>
        
        {resourceMetrics.metrics[0]?.gpu_usage && (
          <Grid item xs={12} md={6}>
            <Paper sx={{ p: 2 }}>
              <Typography variant="h6" gutterBottom>GPU Usage</Typography>
              <ResponsiveContainer width="100%" height={300}>
                <LineChart data={resourceMetrics.metrics}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis 
                    dataKey="timestamp" 
                    tickFormatter={(tick) => new Date(tick * 1000).toLocaleTimeString()}
                    minTickGap={30}
                  />
                  <YAxis domain={[0, 100]} />
                  <Tooltip 
                    formatter={(value) => `${value.toFixed(2)}%`}
                    labelFormatter={(label) => new Date(label * 1000).toLocaleTimeString()}
                  />
                  <Legend />
                  <Line type="monotone" dataKey="gpu_usage[0]" name="GPU 0 Usage (%)" stroke="#ff7300" />
                </LineChart>
              </ResponsiveContainer>
            </Paper>
          </Grid>
        )}
        
        {resourceMetrics.metrics[0]?.gpu_memory && (
          <Grid item xs={12} md={6}>
            <Paper sx={{ p: 2 }}>
              <Typography variant="h6" gutterBottom>GPU Memory</Typography>
              <ResponsiveContainer width="100%" height={300}>
                <LineChart data={resourceMetrics.metrics}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis 
                    dataKey="timestamp" 
                    tickFormatter={(tick) => new Date(tick * 1000).toLocaleTimeString()}
                    minTickGap={30}
                  />
                  <YAxis />
                  <Tooltip 
                    formatter={(value) => `${value.toFixed(2)} GB`}
                    labelFormatter={(label) => new Date(label * 1000).toLocaleTimeString()}
                  />
                  <Legend />
                  <Line type="monotone" dataKey="gpu_memory[0]" name="GPU 0 Memory (GB)" stroke="#ff4500" />
                </LineChart>
              </ResponsiveContainer>
            </Paper>
          </Grid>
        )}
        
        <Grid item xs={12} md={6}>
          <Paper sx={{ p: 2 }}>
            <Typography variant="h6" gutterBottom>Disk I/O</Typography>
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={resourceMetrics.metrics}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis 
                  dataKey="timestamp" 
                  tickFormatter={(tick) => new Date(tick * 1000).toLocaleTimeString()}
                  minTickGap={30}
                />
                <YAxis />
                <Tooltip 
                  formatter={(value) => `${value.toFixed(2)} MB/s`}
                  labelFormatter={(label) => new Date(label * 1000).toLocaleTimeString()}
                />
                <Legend />
                <Line type="monotone" dataKey="disk_read" name="Disk Read (MB/s)" stroke="#8884d8" />
                <Line type="monotone" dataKey="disk_write" name="Disk Write (MB/s)" stroke="#82ca9d" />
              </LineChart>
            </ResponsiveContainer>
          </Paper>
        </Grid>
        
        <Grid item xs={12} md={6}>
          <Paper sx={{ p: 2 }}>
            <Typography variant="h6" gutterBottom>Network I/O</Typography>
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={resourceMetrics.metrics}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis 
                  dataKey="timestamp" 
                  tickFormatter={(tick) => new Date(tick * 1000).toLocaleTimeString()}
                  minTickGap={30}
                />
                <YAxis />
                <Tooltip 
                  formatter={(value) => `${value.toFixed(2)} MB/s`}
                  labelFormatter={(label) => new Date(label * 1000).toLocaleTimeString()}
                />
                <Legend />
                <Line type="monotone" dataKey="network_rx" name="Network Recv (MB/s)" stroke="#8884d8" />
                <Line type="monotone" dataKey="network_tx" name="Network Send (MB/s)" stroke="#82ca9d" />
              </LineChart>
            </ResponsiveContainer>
          </Paper>
        </Grid>
      </Grid>
    </Box>
  );
};

export default SystemResources;