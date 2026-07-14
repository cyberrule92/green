import React, { useContext, useEffect, useState } from 'react';
import {
  Box,
  Button,
  Header,
  ResponsiveContext,
  Text,
  Layer,
  TextInput,
} from 'grommet';
import { Hpe, User } from 'grommet-icons';
import { MenuExample } from './MenuExample';
import { useNavigate } from 'react-router-dom';
import { getTenantId, setTenantId, onTenantChange } from '../lib/api';

export const HeaderExample = ({ onModeratorLogin, onLogout }) => {
  const size = useContext(ResponsiveContext);
  const [showLogin, setShowLogin] = useState(false);
  const [password, setPassword] = useState('');
  const [showTenant, setShowTenant] = useState(false);
  const [activeTenant, setActiveTenant] = useState(getTenantId());
  const [tenantInput, setTenantInput] = useState(getTenantId());
  const [tenantError, setTenantError] = useState('');
  const navigate = useNavigate();

  useEffect(() => onTenantChange((tid) => setActiveTenant(tid)), []);

  const openTenantPicker = () => {
    setTenantInput(activeTenant);
    setTenantError('');
    setShowTenant(true);
  };

  const applyTenant = () => {
    try {
      setTenantId(tenantInput);
      setShowTenant(false);
      // Reload so cached per-tenant data (conversations, observability, etc.)
      // is re-fetched from the new scope.
      window.location.reload();
    } catch (err) {
      setTenantError(err.message || 'Invalid tenant id.');
    }
  };

  const handleLogin = () => {
    if (password === 'secret123') {
      onModeratorLogin(true);
      setShowLogin(false);
      navigate('/moderator');
    } else {
      alert('Wrong password');
    }
  };

  const handleLogout = () => {
    onLogout();
    navigate('/');
  };

  return (
    <Box elevation="small" fill="horizontal">
      <Header
        fill="horizontal"
        pad={{ horizontal: 'medium', vertical: 'small' }}
        background="white"
      >
        {/* Logo + Title (click → Chatbot) */}
        <Button onClick={() => navigate('/')}>
          <Box
            direction="row"
            align="start"
            gap="medium"
            pad={{ vertical: 'xsmall' }}
            responsive={false}
          >
            <Hpe color="green" height="medium" />
            {!['xsmall', 'small'].includes(size) && (
              <Box direction="row" gap="xsmall" wrap>
                <Text color="black" weight="bold">
                  HPE
                </Text>
                <Text color="text-strong">Chatbot</Text>
              </Box>
            )}
          </Box>
        </Button>

        {/* Right-side: tenant chip + menu */}
        <Box direction="row" align="center" gap="small">
          <Button
            plain
            onClick={openTenantPicker}
            tip={`Active tenant: ${activeTenant}. Click to switch.`}
          >
            <Box
              direction="row"
              align="center"
              gap="xsmall"
              pad={{ horizontal: 'small', vertical: 'xsmall' }}
              round="small"
              border={{ color: 'border', size: 'xsmall' }}
              background="background-contrast"
            >
              <User size="small" color="text-strong" />
              {!['xsmall', 'small'].includes(size) && (
                <Text size="small" color="text-strong" weight="bold">
                  {activeTenant}
                </Text>
              )}
            </Box>
          </Button>
          <MenuExample
            items={[
              { label: 'Switch tenant', onClick: openTenantPicker },
              { label: 'Login as Moderator', onClick: () => setShowLogin(true) },
              { label: 'Green AI Chatbot', onClick: () => navigate('/green-ai') },
              { label: 'Logout', onClick: handleLogout },
            ]}
          />
        </Box>

        {/* Tenant switcher Layer */}
        {showTenant && (
          <Layer
            onEsc={() => setShowTenant(false)}
            onClickOutside={() => setShowTenant(false)}
          >
            <Box pad="medium" gap="small" width="medium" background="white" round="small">
              <Text weight="bold" color="text-strong">
                Switch tenant
              </Text>
              <Text size="small" color="text-weak">
                Sets the X-Tenant-Id header on every API call. Conversations,
                RAG documents, budgets, and observability are isolated per
                tenant. Allowed: lowercase, 1–64 chars, [a–z 0–9 _ -].
              </Text>
              <TextInput
                placeholder="default"
                value={tenantInput}
                onChange={(e) => {
                  setTenantInput(e.target.value);
                  setTenantError('');
                }}
                onKeyDown={(e) => {
                  if (e.key === 'Enter') applyTenant();
                }}
              />
              {tenantError && (
                <Text size="small" color="status-critical">
                  {tenantError}
                </Text>
              )}
              <Box direction="row" gap="small" justify="end">
                <Button
                  primary
                  label="Apply"
                  onClick={applyTenant}
                  style={{ backgroundColor: '#00b388', color: 'white' }}
                />
                <Button
                  label="Cancel"
                  onClick={() => setShowTenant(false)}
                  style={{ backgroundColor: '#00b388', color: 'white' }}
                />
              </Box>
            </Box>
          </Layer>
        )}

        {/* Moderator Login Layer */}
        {showLogin && (
          <Layer
            onEsc={() => setShowLogin(false)}
            onClickOutside={() => setShowLogin(false)}
          >
            <Box pad="medium" gap="medium" width="medium" background="white" round="small">
              <Text weight="bold" color="text-strong">
                Moderator Login
              </Text>
              <TextInput
                type="password"
                placeholder="Enter password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
              />
              <Box direction="row" gap="small" justify="end">
                <Button
                  primary
                  label="Login"
                  onClick={handleLogin}
                  style={{ backgroundColor: '#00b388', color: 'white' }} // HPE green
                />
                <Button
                  label="Cancel"
                  onClick={() => setShowLogin(false)}
                  style={{ backgroundColor: '#00b388', color: 'white' }} // HPE green
                />
              </Box>
            </Box>
          </Layer>
        )}
      </Header>
    </Box>
  );
};