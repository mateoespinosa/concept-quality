# Construct the encoder model
import numpy as np
import os
import random
import tensorflow as tf
import yaml
import numpy as np
from importlib import reload
from pathlib import Path
import sklearn
import scipy


############################################################################
## Build input-to-concepts model
############################################################################


def _extract_concepts(activations, concept_cardinality):
    concepts = []
    total_seen = 0
    if all(np.array(concept_cardinality) <= 2):
        # Then nothing to do here as they are all binary concepts
        return activations
    for num_values in concept_cardinality:
        concepts.append(activations[:, total_seen: total_seen + num_values])
        total_seen += num_values
    return concepts
    

def construct_encoder(
    input_shape,
    filter_groups,
    units,
    concept_cardinality,
    drop_prob=0.5,
    max_pool_window=(2,2),
    max_pool_stride=2,
    latent_dims=0,
    output_logits=False,
    sigmoid_growth_rate=1,
):
    encoder_inputs = tf.keras.Input(shape=input_shape)
    encoder_compute_graph = encoder_inputs
    
    # Start with our convolutions
    num_convs = 0
    for filter_group in filter_groups:
        for (num_filters, kernel_size) in filter_group:
            encoder_compute_graph = tf.keras.layers.Conv2D(
                filters=num_filters,
                kernel_size=kernel_size,
                padding="SAME",
                activation=None,
                name=f'encoder_conv_{num_convs}',
            )(encoder_compute_graph)
            num_convs += 1
            encoder_compute_graph = tf.keras.layers.BatchNormalization()(
                encoder_compute_graph
            )
            encoder_compute_graph = tf.keras.activations.relu(encoder_compute_graph)
        # Then do a max pool here to control the parameter count of the model
        # at the end of each group
        encoder_compute_graph = tf.keras.layers.MaxPooling2D(
            pool_size=max_pool_window,
            strides=max_pool_stride,
        )(
            encoder_compute_graph
        )
    
    # Flatten this guy
    encoder_compute_graph = tf.keras.layers.Flatten()(encoder_compute_graph)
    
    # Add a dropout if requested
    if drop_prob:
        encoder_compute_graph = tf.keras.layers.Dropout(drop_prob)(
            encoder_compute_graph
        )
    
    # Finally, include the fully connected bottleneck here
    for i, units in enumerate(units):
        encoder_compute_graph = tf.keras.layers.Dense(
            units,
            activation='relu',
            name=f"encoder_dense_{i}",
        )(encoder_compute_graph)
    
    if latent_dims:
        bypass = tf.keras.layers.Dense(
            latent_dims,
            activation="sigmoid",
            name="encoder_bypass_channel",
        )(encoder_compute_graph)
    else:
        bypass = None
    
    # Map to our output distribution to a flattened
    # vector where we will extract distributions over
    # all concept values
    encoder_compute_graph = tf.keras.layers.Dense(
        sum(concept_cardinality),
        activation=None,
        name="encoder_concept_outputs",
    )(encoder_compute_graph)
        
    # Separate this vector into all of its heads
    concept_outputs = _extract_concepts(
        encoder_compute_graph,
        concept_cardinality,
    )
    if not output_logits:
        if isinstance(concept_outputs, list):
            for i, concept_vec in enumerate(concept_outputs):
                if concept_vec.shape[-1] == 1:
                    # Then this is a binary concept so simply apply sigmoid
                    concept_outputs[i] = 1.0 / (1 + tf.math.exp(- sigmoid_growth_rate * concept_vec))
                else:
                    # Else we will apply a softmax layer as we assume that all of these
                    # entries represent a multi-modal probability distribution
                    concept_outputs[i] = tf.keras.activations.softmax(
                        concept_vec,
                        axis=-1,
                    )
        else:
            # Else they are allbinary concepts so let's sigmoid them
            concept_outputs = tf.keras.activations.sigmoid(concept_outputs)
    return tf.keras.Model(
        encoder_inputs,
        [concept_outputs, bypass] if bypass is not None else concept_outputs,
        name="encoder",
    )

############################################################################
## Build concepts-to-labels model
############################################################################

def construct_decoder(units, num_outputs):
    decoder_layers = [tf.keras.layers.Flatten()] + [
        tf.keras.layers.Dense(
            units,
            activation=tf.nn.relu,
            name=f"decoder_dense_{i+1}",
        ) for i, units in enumerate(units)
    ]
    return tf.keras.Sequential(decoder_layers + [
        tf.keras.layers.Dense(
            num_outputs if num_outputs > 2 else 1,
            activation=None,
            name="decoder_model_output",
        )
    ])



############################################################################
## Build end-to-end model
############################################################################


# Construct the complete model
def construct_end_to_end_model(
    input_shape,
    encoder,
    decoder,
    num_outputs,
    learning_rate=1e-3,
):
    model_inputs = tf.keras.Input(shape=input_shape)
    latent = encoder(model_inputs)
    if isinstance(latent, list):
        if len(latent) > 1:
            compacted_vector = tf.keras.layers.Concatenate(axis=-1)(
                latent
            )
        else:
            compacted_vector = latent[0]
    else:
        compacted_vector = latent
    model_compute_graph = decoder(compacted_vector)
    # Now time to collapse all the concepts again back into a single vector
    model = tf.keras.Model(
        model_inputs,
        model_compute_graph,
        name="complete_model",
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        loss=(
            tf.keras.losses.BinaryCrossentropy(from_logits=True) if (num_outputs <= 2)
            else tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        ),
        metrics=[
            "binary_accuracy" if (num_outputs <= 2)
            else tf.keras.metrics.SparseTopKCategoricalAccuracy(k=1)
        ],
    )
    return model, encoder, decoder